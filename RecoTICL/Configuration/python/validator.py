# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Type-aware plumbing validation.

After a :class:`~RecoTICL.Configuration.model.TICLConfig` is assembled into ``cms``
modules, the validator builds the *product graph* (every product each module
puts into the event, keyed by ``(module_label, instance_label)`` with its C++
type) and checks every consumed ``InputTag``:

* the referenced module is one pyTICL builds (else it is treated as an external
  upstream input -- e.g. ``hgcalMergeLayerClusters``, ``generalTracks``);
* it produces a product with the requested instance label;
* the produced C++ type matches the type the consumer requires.

It also checks plugin-type strings against the catalog enums and rejects a GPU
backend request on a module that has no GPU (alpaka) implementation.

Connections that cannot type-check raise :class:`PyTICLError` with a precise,
actionable message.
"""

import FWCore.ParameterSet.Config as cms

from RecoTICL.Configuration.catalog import (
    CATALOG, GPU,
    SEEDING_TYPES, FILTER_TYPES, PATTERN_TYPES,
)
from RecoTICL.Configuration.model import PyTICLError
from RecoTICL.Configuration.assembler import trackster_label, filter_label


def _input_tags(value):
    """Return ``[(moduleLabel, instanceLabel), ...]`` for an InputTag/VInputTag."""
    tags = []
    if isinstance(value, cms.InputTag):
        tags.append((value.getModuleLabel(), value.getProductInstanceLabel()))
    elif isinstance(value, cms.VInputTag):
        for el in value:
            if isinstance(el, cms.InputTag):
                tags.append((el.getModuleLabel(), el.getProductInstanceLabel()))
            else:  # string form "module:instance"
                parts = str(el).split(":")
                tags.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return tags


def _product_graph(modules):
    """``{(label, instance): {cpp_type, ...}}`` for every product of every module.

    A module may put several products into the event under the *same* instance
    label (typically the empty one), distinguished only by their C++ type -- so
    each ``(label, instance)`` maps to a *set* of types.
    """
    produced = {}
    for label, mod in modules.items():
        spec = CATALOG.get(mod.type_())
        if spec is None:
            continue
        for p in spec.produces:
            produced.setdefault((label, p.instance_label(mod)), set()).add(p.cpp_type)
    return produced


def validate(cfg):
    """Validate ``cfg``; raise :class:`PyTICLError` on any problem, else return the
    assembled result."""
    assembled = cfg.assemble()
    modules = assembled.modules
    produced = _product_graph(modules)
    errors = []

    # --- type-aware connection checks ----------------------------------- #
    for label, mod in modules.items():
        spec = CATALOG.get(mod.type_())
        if spec is None:
            continue
        for c in spec.consumes:
            if not hasattr(mod, c.param):
                continue
            for tmod, tinst in _input_tags(getattr(mod, c.param)):
                if tmod == "" or tmod not in modules:
                    continue  # unset, or an external upstream input -- allowed
                want = c.cpp_type
                got = produced.get((tmod, tinst))
                if got is None:
                    avail = sorted(i for (m, i) in produced if m == tmod)
                    errors.append(
                        "%s.%s -> %s:%s is not produced (%s produces instances %s)"
                        % (label, c.param, tmod, tinst or "<none>", tmod,
                           [a or "<none>" for a in avail]))
                elif want not in got:
                    errors.append(
                        "%s.%s type mismatch: needs %s but %s:%s produces %s"
                        % (label, c.param, want, tmod, tinst or "<none>",
                           ", ".join(sorted(got))))

    # --- plugin type-string checks -------------------------------------- #
    for it in cfg.iterations:
        if it.seeding_type is not None and it.seeding_type not in SEEDING_TYPES:
            errors.append("iteration %r: unknown seeding type %r" % (it.name, it.seeding_type))
        if it.filter_type is not None and it.filter_type not in FILTER_TYPES:
            errors.append("iteration %r: unknown cluster filter %r" % (it.name, it.filter_type))
        if it.pattern_type is not None and it.pattern_type not in PATTERN_TYPES:
            errors.append("iteration %r: unknown pattern recognition %r" % (it.name, it.pattern_type))

    # --- backend checks (GPU only where supported) ---------------------- #
    for it in cfg.iterations:
        if it.backend != GPU:
            continue
        for lbl in (filter_label(it.name), trackster_label(it.name)):
            mod = modules.get(lbl)
            if mod is None:
                continue
            spec = CATALOG.get(mod.type_())
            if spec is not None and not spec.supports(GPU):
                errors.append(
                    "iteration %r requests GPU but module %s (%s) has no GPU "
                    "(alpaka) implementation; available backends: %s"
                    % (it.name, lbl, mod.type_(), ", ".join(spec.backends)))

    if errors:
        raise PyTICLError(
            "pyTICL plumbing validation failed (%d problem(s)):\n  - %s"
            % (len(errors), "\n  - ".join(errors)))
    return assembled
