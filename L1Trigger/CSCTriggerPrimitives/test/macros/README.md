The macro CCLUTLinearFitWriter.cpp produces look-up tables for the Run-3 CSC trigger.

* 10 LUTs will be created for CMSSW (5 for position offset, 5 for slope). 5 LUTs will be created for the Verilog firmware. These  files are similar to the ones found here: https://github.com/cms-data/L1Trigger-CSCTriggerPrimitives. 5 additional LUTs will be created that convert the Run-3 comparator code & pattern ID to a Run-1/2 pattern ID. The LUTs used in the simulation require at least 3 layers. For reference the macro also produces fits with at least four layers.

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
