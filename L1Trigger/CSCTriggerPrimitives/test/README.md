CSC Trigger Primitives: Test Modules
====================================

runCSCTriggerPrimitiveProducer
------------------------------

Configuration to run the CSC trigger primitive producer. Option available to unpack RAW data - useful for data vs emulator comparisons.


runCSCL1TDQMClient
------------

Configuration to run the module `l1tdeCSCTPGClient` on a data file. Will produce a DQM file onto which the runCSCTriggerPrimitiveAnalyzer can be run


runCSCTriggerPrimitiveAnalyzer
------------------------------

Configuration to run analysis on CSC trigger primitives. Choose from options to analyze data vs emulator comparison, MC resolution or MC efficiency.

For data vs emulator comparison, first run `runCSCTriggerPrimitiveProducer` on RAW data with the dqm enabled, followed by `runCSCL1TDQMClient`.


runL1CSCTPEmulatorConfigAnalyzer
--------------------------------

Compare configuration from DB with Python for CSC trigger primitives. Typically not necessary to do; all configuration is by default loaded from Python, not the configuration DB.


runGEMCSCLUTAnalyzer
--------------------

Makes the lookup tables for the GEM-CSC integrated local trigger in simulation and firmware. Current lookup tables can be found at https://github.com/cms-data/L1Trigger-CSCTriggerPrimitives/tree/master/GEMCSC


CCLUTLinearFitWriter
--------------------

The macro CCLUTLinearFitWriter.cpp produces look-up tables for the Run-3 CSC trigger using the comparator code logic.

* 10 LUTs will be created for CMSSW (5 for position offset, 5 for slope). 5 LUTs will be created for the Verilog firmware. 5 additional LUTs will be created that convert the Run-3 comparator code & pattern ID to a Run-1/2 pattern ID. The LUTs used in the simulation require at least 3 layers. The macro also produces fits with at least four layers.

<PRE>
mkdir output_3layers
mkdir output_4layers
mkdir figures_3layers
mkdir figures_4layers
root -l -q -b CCLUTLinearFitWriter.cpp++(3)
root -l -q -b CCLUTLinearFitWriter.cpp++(4)
</PRE>

* Convention for 4-bit position offset word:

| Value | Half-Strip Offset  | Delta Half-Strip  | Quarter-Strip Bit  | Eighth-Strip Bit |
|-------|--------------------|-------------------|--------------------|------------------|
|   0   |   -7/4             |   -2              |   0                |   1              |
|   1   |   -3/2             |   -2              |   1                |   0              |
|   2   |   -5/4             |   -2              |   1                |   1              |
|   3   |   -1               |   -1              |   0                |   0              |
|   4   |   -3/4             |   -1              |   0                |   1              |
|   5   |   -1/2             |   -1              |   1                |   0              |
|   6   |   -1/4             |   -1              |   1                |   1              |
|   7   |   0                |   0               |   0                |   0              |
|   8   |   1/4              |   0               |   0                |   1              |
|   9   |   1/2              |   0               |   1                |   0              |
|   10  |   3/4              |   0               |   1                |   1              |
|   11  |   1                |   1               |   0                |   0              |
|   12  |   5/4              |   1               |   0                |   1              |
|   13  |   3/2              |   1               |   1                |   0              |
|   14  |   7/4              |   1               |   1                |   1              |
|   15  |   2                |   2               |   0                |   0              |

* Convention for 4-bit slope:

Slope is in units [half-strip offset/layer]. The sign of the bending is interpreted as in Run-1 and Run-2.

| Value | Slope |
|-------|-------|
|   0   |  1/8  |
|   1   |  2/8  |
|   2   |  3/8  |
|   3   |  4/8  |
|   4   |  5/8  |
|   5   |  6/8  |
|   6   |  7/8  |
|   7   |   1   |
|   8   |  9/8  |
|   9   | 10/8  |
|  10   | 11/8  |
|  11   | 12/8  |
|  12   | 13/8  |
|  13   | 14/8  |
|  14   |   2   |
|  15   | 20/8  |

The LUTs can be found at https://github.com/cms-data/L1Trigger-CSCTriggerPrimitives/tree/master/CCLUT
