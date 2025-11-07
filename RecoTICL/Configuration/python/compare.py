# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Helpers to compare an assembled TICL config against a baseline ``cms.Task``.

The comparison is *per-module*: the set of leaf module labels in a task, plus
each module's :meth:`dumpPython` keyed by label.  This is robust to module
sharing and to cosmetic differences in how sub-tasks are nested.
"""

import difflib


def normalized_task(process, task):
    """Return ``(labels, {label: dumpPython})`` for every leaf module in ``task``."""
    labels = sorted(task.moduleNames())
    dumps = {l: getattr(process, l).dumpPython() for l in labels}
    return labels, dumps


def diff_tasks(baseline_process, baseline_task, test_process, test_task):
    """Return a human-readable diff string, or ``""`` if the tasks are identical."""
    base_labels, base_dumps = normalized_task(baseline_process, baseline_task)
    test_labels, test_dumps = normalized_task(test_process, test_task)

    out = []
    base_set, test_set = set(base_labels), set(test_labels)
    if base_set != test_set:
        missing = sorted(base_set - test_set)
        extra = sorted(test_set - base_set)
        if missing:
            out.append("MISSING modules (in baseline, not generated): %s" % ", ".join(missing))
        if extra:
            out.append("EXTRA modules (generated, not in baseline): %s" % ", ".join(extra))

    for label in sorted(base_set & test_set):
        if base_dumps[label] != test_dumps[label]:
            d = difflib.unified_diff(
                base_dumps[label].splitlines(),
                test_dumps[label].splitlines(),
                fromfile="baseline/%s" % label,
                tofile="pyTICL/%s" % label,
                lineterm="",
            )
            out.append("MODULE DIFFERS: %s\n%s" % (label, "\n".join(d)))

    return "\n".join(out)
