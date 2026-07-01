# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Graphviz (DOT) export and interactive viewing of an assembled pyTICL graph.

The graph is the module dependency graph of an assembled ``TICLConfig``:

* nodes are the modules pyTICL builds -- their python labels, the EDProducer
  type, and the scalar configuration parameters -- plus the external upstream
  inputs they consume (e.g. ``hgcalMergeLayerClusters``, ``generalTracks``);
* edges carry the product that flows from a producer to a consumer, labelled
  with the consumer's parameter, the C++ product type and the instance label.

Usage::

    cfg.to_dot("ticl.dot")   # write a Graphviz DOT file (also returns the string)
    cfg.show_graph()         # render and open it, or inline in a Jupyter notebook
"""

import os
import re
import sys
import tempfile

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import CATALOG
from RecoTICL.Configuration.validator import _input_tags

# scalar / short-vector cms parameter types surfaced inside a node label
_SCALAR = (cms.int32, cms.uint32, cms.int64, cms.uint64, cms.double, cms.bool, cms.string)
_VSCALAR = (cms.vint32, cms.vuint32, cms.vdouble, cms.vstring)


def _as_assembled(cfg_or_assembled):
    """Accept either a ``TICLConfig`` (assemble it) or an already-assembled result."""
    if hasattr(cfg_or_assembled, "assemble"):
        return cfg_or_assembled.assemble()
    return cfg_or_assembled


def _node_config(mod, max_params=10, max_vec=4):
    """Compact ``key = value`` strings for a module's scalar / short-vector params."""
    rows = []
    for k in sorted(mod.parameterNames_()):
        try:
            v = getattr(mod, k)
        except Exception:
            continue
        if isinstance(v, _SCALAR):
            rows.append("%s = %s" % (k, v.value()))
        elif isinstance(v, _VSCALAR):
            vals = list(v)
            if 0 < len(vals) <= max_vec:
                rows.append("%s = %s" % (k, vals))
    return rows[:max_params]


def build_graph(assembled):
    """Build the module dependency graph of an assembled config.

    Returns ``(nodes, edges)`` where

    * ``nodes`` is ``{label: {"type": str, "kind": "module"|"external",
      "config": [str, ...]}}``;
    * ``edges`` is ``[{"src", "dst", "param", "cpp_type", "instance"}, ...]``,
      one per consumed ``InputTag`` resolved to a module pyTICL builds (or an
      external upstream input).
    """
    modules = assembled.modules
    nodes = {}
    edges = []

    for label, mod in modules.items():
        nodes[label] = {"type": mod.type_(), "kind": "module", "config": _node_config(mod)}

    for label, mod in modules.items():
        spec = CATALOG.get(mod.type_())
        if spec is None:
            continue
        for c in spec.consumes:
            if not hasattr(mod, c.param):
                continue
            for tmod, tinst in _input_tags(getattr(mod, c.param)):
                if tmod == "":
                    continue  # unset InputTag
                if tmod not in nodes:
                    nodes[tmod] = {"type": "external input", "kind": "external", "config": []}
                edges.append({"src": tmod, "dst": label, "param": c.param,
                              "cpp_type": c.cpp_type, "instance": tinst})
    return nodes, edges


def _short_type(t):
    """Drop C++ namespaces for readability (``std::vector<ticl::Trackster>`` ->
    ``vector<Trackster>``)."""
    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*::", "", t)


def _esc(s):
    return str(s).replace("\\", "\\\\").replace('"', '\\"')


def to_dot(cfg_or_assembled, path=None, show_config=True, short_types=True):
    """Build the Graphviz DOT of the assembled module graph.

    ``show_config`` adds each module's scalar parameters to its node; ``short_types``
    strips C++ namespaces from the edge labels.  Writes the DOT to ``path`` when
    given; always returns the DOT string.
    """
    assembled = _as_assembled(cfg_or_assembled)
    nodes, edges = build_graph(assembled)

    out = [
        "digraph pyTICL {",
        "  rankdir=LR;",
        '  graph [fontname="monospace"];',
        '  node [shape=box, fontname="monospace", fontsize=10, style=filled, fillcolor="#e8f0fe"];',
        '  edge [fontname="monospace", fontsize=8, color="#555555"];',
    ]
    for label in sorted(nodes):
        info = nodes[label]
        if info["kind"] == "external":
            out.append('  "%s" [label="%s\\n(external input)", shape=ellipse, fillcolor="#eeeeee"];'
                       % (_esc(label), _esc(label)))
        else:
            rows = [label, "<%s>" % info["type"]]
            if show_config:
                rows += info["config"]
            out.append('  "%s" [label="%s"];' % (_esc(label), "\\n".join(_esc(r) for r in rows)))
    for e in sorted(edges, key=lambda x: (x["src"], x["dst"], x["param"])):
        typ = _short_type(e["cpp_type"]) if short_types else e["cpp_type"]
        inst = (":" + e["instance"]) if e["instance"] else ""
        lab = "%s\\n%s%s" % (e["param"], typ, inst)
        out.append('  "%s" -> "%s" [label="%s"];' % (_esc(e["src"]), _esc(e["dst"]), _esc(lab)))
    out.append("}")
    dot = "\n".join(out) + "\n"

    if path:
        with open(path, "w") as fh:
            fh.write(dot)
    return dot


def _in_notebook():
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and getattr(ip, "kernel", None) is not None
    except Exception:
        return False


def _open_file(path):
    import subprocess
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)  # noqa: B606
        else:
            subprocess.Popen(["xdg-open", path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def show_graph(cfg_or_assembled, filename=None, fmt="svg", view=True, **dot_opts):
    """Render the module graph and show it interactively.

    * In a Jupyter notebook, returns a ``graphviz.Source`` for inline display.
    * Otherwise renders to ``fmt`` (svg/png/pdf) and, if ``view``, opens it with
      the platform viewer.
    * If neither the ``graphviz`` python package nor the ``dot`` binary is
      available, writes the DOT and prints how to render it.

    Returns the path of the produced artifact (or the ``Source`` in a notebook).
    """
    dot = to_dot(cfg_or_assembled, **dot_opts)
    stem = filename or os.path.join(tempfile.gettempdir(), "pyticl_graph")

    # 1) the graphviz python package: inline in notebooks, rendered elsewhere
    try:
        import graphviz
        src = graphviz.Source(dot, format=fmt)
        if _in_notebook() and filename is None:
            return src
        rendered = src.render(filename=stem, view=view and not _in_notebook(), cleanup=True)
        print("pyTICL graph: %s" % rendered)
        return rendered
    except ImportError:
        pass

    # 2) the `dot` command-line tool
    import shutil
    import subprocess
    dot_path = stem + ".dot"
    with open(dot_path, "w") as fh:
        fh.write(dot)
    if shutil.which("dot"):
        img = "%s.%s" % (stem, fmt)
        subprocess.run(["dot", "-T%s" % fmt, dot_path, "-o", img], check=True)
        if view:
            _open_file(img)
        print("pyTICL graph: %s" % img)
        return img

    # 3) nothing to render with: leave the DOT and explain
    print("graphviz not available; wrote DOT to %s\n  render it with:  dot -T%s %s -o graph.%s"
          % (dot_path, fmt, dot_path, fmt))
    return dot_path
