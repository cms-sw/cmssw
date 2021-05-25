# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

### Load all default matrices (see Configuration/PyReleaseValidation/python/MatrixReader.py)
from Configuration.PyReleaseValidation.relval_standard import workflows as _standard
from Configuration.PyReleaseValidation.relval_highstats import workflows as _highstats
from Configuration.PyReleaseValidation.relval_pileup import workflows as _pileup
from Configuration.PyReleaseValidation.relval_generator import workflows as _generator
from Configuration.PyReleaseValidation.relval_extendedgen import workflows as _extendedgen
from Configuration.PyReleaseValidation.relval_production import workflows as _production
from Configuration.PyReleaseValidation.relval_ged import workflows as _ged
from Configuration.PyReleaseValidation.relval_gpu import workflows as _gpu
from Configuration.PyReleaseValidation.relval_2017 import workflows as _2017
from Configuration.PyReleaseValidation.relval_2026 import workflows as _2026
from Configuration.PyReleaseValidation.relval_machine import workflows as _machine
from Configuration.PyReleaseValidation.relval_premix import workflows as _premix

from Configuration.PyReleaseValidation.relval_upgrade import workflows as _upgrade_workflows

numWFIB = []
for upgrade_wf in _upgrade_workflows:
    veto = False
    for matrixToVeto in [_standard, _highstats, _pileup, _generator, _extendedgen, _production, _ged, _gpu, _2017, _2026, _machine, _premix]:
        if upgrade_wf in matrixToVeto:
            veto = True
            break
    if not veto:
        numWFIB.extend([upgrade_wf])

for numWF in numWFIB:
    workflows[numWF] = _upgrade_workflows[numWF]
