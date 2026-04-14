#!/usr/bin/env python3
"""
Strict XML validator using Python's xml.etree.ElementTree parser.
Reads XML from stdin and exits with code 0 if valid, non-zero if invalid.
"""
import sys
import xml.etree.ElementTree as ET

try:
    xml_content = sys.stdin.read()
    ET.fromstring(xml_content)
    sys.exit(0)
except ET.ParseError as e:
    print(f"XML validation failed: {e}", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("XML content:", file=sys.stderr)
    print(xml_content, file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error during XML validation: {e}", file=sys.stderr)
    sys.exit(2)
