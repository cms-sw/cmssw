# coding: utf-8

"""
AOT compilation and dev workflow tests.
"""

import os
import sys
import shlex
import subprocess
import tempfile
import functools
import unittest


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
    def test_compilation(self, tmp_dir):
        # create the test model
        cmd = [
            sys.executable,
            "-W", "ignore",
            os.path.join(this_dir, "create_model.py"),
            "-d", os.path.join(tmp_dir, "testmodel"),
        ]
        run_cmd(cmd)

        # compile it
        cmd = [
            "PYTHONWARNINGS=ignore",
            "cmsml_compile_tf_graph",
            os.path.join(tmp_dir, "testmodel"),
            os.path.join(tmp_dir, "testmodel_compiled"),
            "-c", r"testmodel_bs{}", r"testmodel_bs{}",
            "-b", "1,2",
        ]
        run_cmd(cmd)

        # check files
        exists = lambda *p: os.path.exists(os.path.join(tmp_dir, "testmodel_compiled", "aot", *p))
        self.assertTrue(exists("testmodel_bs1.h"))
        self.assertTrue(exists("testmodel_bs1.o"))
        self.assertTrue(exists("testmodel_bs2.h"))
        self.assertTrue(exists("testmodel_bs2.o"))

    @run_in_tmp
    def test_dev_workflow(self, tmp_dir):
        # create the test model
        cmd = [
            sys.executable,
            "-W", "ignore",
            os.path.join(this_dir, "create_model.py"),
            "-d", os.path.join(tmp_dir, "testmodel"),
        ]
        run_cmd(cmd)

        # compile it
        cmd = [
            sys.executable,
            "-W", "ignore",
            os.path.normpath(os.path.join(this_dir, "..", "scripts", "compile_model.py")),
            "-m", os.path.join(tmp_dir, "testmodel"),
            "-s", "PhysicsTools",
            "-p", "TensorFlowAOT",
            "-b", "1,2",
            "-o", os.path.join(tmp_dir, "testmodel_compiled"),
        ]
        run_cmd(cmd)

        # check files
        exists = lambda *p: os.path.exists(os.path.join(tmp_dir, "testmodel_compiled", *p))
        self.assertTrue(exists("tfaot-dev-physicstools-tensorflowaot-testmodel.xml"))
        self.assertTrue(exists("include", "testmodel.h"))
        self.assertTrue(exists("include", "testmodel_bs1.h"))
        self.assertTrue(exists("include", "testmodel_bs2.h"))
        self.assertTrue(exists("lib", "testmodel_bs1.o"))
        self.assertTrue(exists("lib", "testmodel_bs2.o"))


if __name__ == "__main__":
    unittest.main()
