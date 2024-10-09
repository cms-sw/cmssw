# coding: utf-8

"""
AOT compilation and dev workflow tests.
"""

import os
import re
import shlex
import subprocess
import tempfile
import functools
import unittest
import platform


this_dir = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd):
    if not isinstance(cmd, str):
        cmd = shlex.join(cmd)
    return subprocess.run(cmd, shell=True, check=True)


def run_in_tmp(func):
    @functools.wraps(func)
    def wrapper(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            func(self, tmp_dir)
    return wrapper


class TFAOTTests(unittest.TestCase):

    @run_in_tmp
    def test_dev_workflow(self, tmp_dir):
        import cms_tfaot

        # find the cms_tfaot install dir to locate the test model
        m = re.match(r"(.+/\d+\.\d+\.\d+\-[^/]+)/lib/.+$", cms_tfaot.__file__)
        self.assertIsNotNone(m)
        config_file = os.path.join(m.group(1), "share", "test_models", "simple", "aot_config.yaml")
        self.assertTrue(os.path.exists(config_file))

        arch = "{0}-pc-linux".format(platform.processor())

        # run the dev workflow
        # create the test model
        cmd = [
            "cms_tfaot_compile",
            "-c", config_file,
            "-o", tmp_dir,
            "--tool-name", "tfaot-model-test",
            "--dev",
            "--additional-flags=--target_triple=" + arch
        ]
        run_cmd(cmd)

        # check files
        exists = lambda *p: os.path.exists(os.path.join(tmp_dir, *p))
        self.assertTrue(exists("tfaot-model-test.xml"))
        self.assertTrue(exists("include", "tfaot-model-test"))
        self.assertTrue(exists("include", "tfaot-model-test", "test_simple_bs1.h"))
        self.assertTrue(exists("include", "tfaot-model-test", "test_simple_bs2.h"))
        self.assertTrue(exists("include", "tfaot-model-test", "model.h"))
        self.assertTrue(exists("lib", "test_simple_bs1.o"))
        self.assertTrue(exists("lib", "test_simple_bs2.o"))


if __name__ == "__main__":
    unittest.main()
