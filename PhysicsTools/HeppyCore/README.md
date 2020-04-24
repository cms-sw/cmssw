Heppy : a python framework for high-energy physics data analysis
================================================================

This README is for the standalone version of heppy.

Prerequisites
-------------

**python 2.x, x>5**

**ROOT 5, with pyroot support**

Note that you need to ensure that ROOT was compiled with the same
version of python as the one in your PATH.

To check that, do the following:

    python
    import ROOT

Any error message needs to be taken care of before going further. 

Environment
-----------
Put the heppy package in a directory that is in your PYTHONPATH
For example, you can do:

    export PYTHONPATH=$PWD/..:$PYTHONPATH

Check that you can now import heppy:

    python
    import heppy 

From this directory, run the initialization script, which makes a few
executable scripts available to you:

    source init.sh


Examples
--------

A simple example are provided in the test/ directory:

    cd test/

Create a root file with a tree:

    python create_tree.py
	
Process the root file:

    multiloop.py  Output   simple_example_cfg.py

Investigate the contents of the Output folder and its subdirectories. 
