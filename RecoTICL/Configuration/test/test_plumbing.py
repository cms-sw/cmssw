#!/usr/bin/env python3
# Original Author: Felice Pantaleo, CERN, felice.pantaleo@cern.ch
"""Positive + negative plumbing/validation tests.

Confirms the type-aware validator accepts the v5 default and rejects:
  * a GPU backend request on a CPU-only module;
  * an iteration whose mask source does not exist;
  * an unknown seeding type.
"""

import sys

from RecoTICL.Configuration.model import TICLConfig, Global, PyTICLError
from RecoTICL.Configuration import presets


def expect_ok(name, fn):
    try:
        fn()
        print("OK   : %s" % name)
        return True
    except Exception as exc:  # noqa: BLE001
        print("FAIL : %s -- unexpected error: %s" % (name, exc))
        return False


def expect_raises(name, fn, needle=None):
    try:
        fn()
    except PyTICLError as exc:
        if needle and needle not in str(exc):
            print("FAIL : %s -- raised but message lacked %r: %s" % (name, needle, exc))
            return False
        print("OK   : %s (correctly rejected)" % name)
        return True
    print("FAIL : %s -- expected PyTICLError, none raised" % name)
    return False


def main():
    ok = True

    # positive: the v5 default validates cleanly
    ok &= expect_ok("v5 default validates", lambda: presets.v5().validate())

    # negative: GPU requested on a CPU-only iteration
    ok &= expect_raises(
        "GPU on CPU-only module rejected",
        lambda: TICLConfig("g").iteration("CLUE3DHigh").preset().on_gpu().validate(),
        needle="no GPU",
    )

    # negative: masks_from an iteration that doesn't exist
    ok &= expect_raises(
        "masks_from unknown iteration rejected",
        lambda: (TICLConfig("m").iteration("Recovery").preset()
                 .masks_from("DoesNotExist").validate()),
        needle="unknown iteration",
    )

    # negative: unknown seeding type rejected at build time
    def bad_seeding():
        TICLConfig("s").iteration("X").seeding("SeedingRegionNope")
    ok &= expect_raises("unknown seeding type rejected", bad_seeding, needle="unknown seeding")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
