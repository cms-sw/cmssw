#!/usr/bin/env python3

import unittest

from Configuration.DataProcessing.Merge import mergeProcess


class MergeTest(unittest.TestCase):
    def test_module_types(self):
        for options, source_type, output_type in (
            ({}, "PoolSource", "PoolOutputModule"),
            ({"rntuple": True}, "RNTupleTempSource", "RNTupleTempOutputModule"),
            ({"mergeNANO": True}, "PoolSource", "NanoAODOutputModule"),
            ({"mergeNANO": True, "rntuple": True}, "RNTupleTempSource", "NanoAODRNTupleOutputModule"),
            ({"newDQMIO": True, "rntuple": True}, "DQMRootSource", "DQMRootOutputModule"),
            ({"mergeNANO": True, "isL1Scouting": True, "rntuple": True}, "RNTupleTempSource", "OrbitNanoAODOutputModule"),
        ):
            with self.subTest(options=options):
                process = mergeProcess("file:input.root", output_file="merged.root", **options)
                self.assertEqual(process.source.type_(), source_type)
                self.assertEqual(process.Merged.type_(), output_type)
                self.assertEqual(list(process.source.fileNames), ["file:input.root"])
                self.assertEqual(process.Merged.fileName.value(), "merged.root")


if __name__ == "__main__":
    unittest.main()
