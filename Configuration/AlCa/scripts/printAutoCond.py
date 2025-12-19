#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Usage examples:
#   python printAutoCond.py --list
#   python printAutoCond.py --key run2_design
#   python printAutoCond.py --pattern run2 --json
#
from __future__ import print_function

import os
import sys
import argparse
import logging
import pprint
import json

logger = logging.getLogger("getAutoCond")


def get_cmssw_release():
    """Return the CMSSW release name from environment or exit with error."""
    release = os.environ.get("CMSSW_VERSION")
    if not release:
        logger.error("CMSSW not properly set. Please source a CMSSW environment (e.g. cmsenv).")
        sys.exit(2)
    return release


def load_autoCond():
    """Import Configuration.AlCa.autoCond and return autoCond dict.

    Exits with non-zero code if import fails.
    """
    try:
        # Lazy import so error message is meaningful
        from Configuration.AlCa.autoCond import autoCond  # type: ignore
    except Exception as exc:
        logger.error("Failed to import Configuration.AlCa.autoCond: %s", exc)
        sys.exit(3)
    if not isinstance(autoCond, dict):
        logger.error("Imported autoCond is not a dict.")
        sys.exit(4)
    return autoCond


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Inspect Configuration.AlCa.autoCond mapping and extract GlobalTags (GTs)."
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "-k",
        "--key",
        dest="key",
        help="Exact key to show the GT for (e.g. 'run2_design')",
    )
    group.add_argument(
        "-l",
        "--list",
        dest="list_keys",
        action="store_true",
        help="List available keys",
    )
    p.add_argument(
        "-p",
        "--pattern",
        dest="pattern",
        help="Filter keys by substring (case-sensitive)",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output result as JSON (useful for scripting)",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return p.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    args = parse_args(argv)

    # Configure logging
    handler = logging.StreamHandler()
    fmt = "%(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)

    release = get_cmssw_release()
    logger.debug("CMSSW release: %s", release)

    autoCond = load_autoCond()

    # Build filtered view
    keys = sorted(autoCond.keys())

    if args.pattern:
        keys = [k for k in keys if args.pattern in k]

    if args.list_keys or (not args.key and not args.pattern):
        # When --list is given OR no specific key/pattern provided, show keys (or filtered keys)
        if args.as_json:
            print(json.dumps({"keys": keys}, indent=2))
            return 0
        for k in keys:
            print(k)
        # If the user asked only to list, exit successfully
        if args.list_keys or not args.key:
            return 0

    # If exact key requested
    if args.key:
        if args.key in autoCond:
            value = autoCond[args.key]
            if args.as_json:
                print(json.dumps({args.key: value}, indent=2, default=str))
            else:
                # Pretty print nested structures
                pp = pprint.PrettyPrinter(depth=6)
                pp.pprint({args.key: value})
            return 0
        else:
            logger.error("Requested key '%s' not found. Use --list to see available keys.", args.key)
            return 5

    # If pattern selected: print mapping for matched keys
    if args.pattern:
        matched = {k: autoCond[k] for k in keys}
        if args.as_json:
            print(json.dumps(matched, indent=2, default=str))
        else:
            pp = pprint.PrettyPrinter(depth=6)
            pp.pprint(matched)
        return 0

    # Fallback: print whole autoCond
    if args.as_json:
        print(json.dumps(autoCond, indent=2, default=str))
    else:
        pp = pprint.PrettyPrinter(depth=6)
        pp.pprint(autoCond)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(130)
