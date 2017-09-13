'''Example configuration file for an ee->ZH->mumubb analysis in heppy, with the FCC-ee

While studying this file, open it in ipython as well as in your editor to 
get more information: 

ipython
from analysis_ee_ZH_cfg import * 

'''

import os
import copy
import PhysicsTools.HeppyCore.framework.config as cfg
import PhysicsTools.HeppyCore.utils.pdebug

import logging
# next 2 lines necessary to deal with reimports from ipython
logging.shutdown()
reload(logging)
logging.basicConfig(level=logging.WARNING)

# setting the random seed for reproducible results
import PhysicsTools.HeppyCore.statistics.rrandom as random
random.seed(0xdeadbeef)

from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events
import PhysicsTools.HeppyCore.utils.pdebug

# definition of the collider
from PhysicsTools.HeppyCore.configuration import Collider
Collider.BEAMS = 'ee'
Collider.SQRTS = 240.

# input definition
comp = cfg.Component(
    'ee_ZH_Zmumu_Hbb',
    files = [
        'ee_ZH_Zmumu_Hbb.root'
    ]
)
selectedComponents = [comp]

# read FCC EDM events from the input root file(s)
# do help(Reader) for more information
from PhysicsTools.HeppyCore.analyzers.fcc.Reader import Reader
source = cfg.Analyzer(
    Reader,
    gen_particles = 'GenParticle',
    gen_vertices = 'GenVertex'
)

from PhysicsTools.HeppyCore.test.papas_cfg import gen_particles_stable

# Use a Filter to select leptons from the output of papas simulation.
# Currently, we're treating electrons and muons transparently.
# we could use two different instances for the Filter module
# to get separate collections of electrons and muons
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
leptons_true = cfg.Analyzer(
    Filter,
    'sel_leptons',
    output = 'leptons_true',
    input_objects = 'gen_particles_stable',
    filter_func = lambda ptc: ptc.e()>10. and abs(ptc.pdgid()) in [11, 13]
)

# Compute lepton isolation w/r other particles in the event.
# help(IsolationAnalyzer) for more information
from PhysicsTools.HeppyCore.analyzers.IsolationAnalyzer import IsolationAnalyzer
from PhysicsTools.HeppyCore.particles.isolation import EtaPhiCircle
iso_leptons = cfg.Analyzer(
    IsolationAnalyzer,
    leptons = 'leptons_true',
    particles = 'gen_particles_stable',
    iso_area = EtaPhiCircle(0.4),
    log_level = logging.INFO
)

# definition of a sequence of analyzers,
# the analyzers will process each event in this order
sequence = cfg.Sequence(
    source,
    gen_particles_stable,
    leptons_true,
    iso_leptons,
)

# Specifics to read FCC events 
from ROOT import gSystem
gSystem.Load("libdatamodelDict")
from EventStore import EventStore as Events

config = cfg.Config(
    components = selectedComponents,
    sequence = sequence,
    services = [],
    events_class = Events
)

