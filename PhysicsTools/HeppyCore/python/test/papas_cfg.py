import PhysicsTools.HeppyCore.framework.config as cfg
from PhysicsTools.HeppyCore.configuration import Collider

# Use a Filter to select stable gen particles for simulation
# from the output of "source" 
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
gen_particles_stable = cfg.Analyzer(
    Filter,
    output = 'gen_particles_stable',
    # output = 'particles',
    input_objects = 'gen_particles',
    filter_func = lambda x : x.status()==1 and abs(x.pdgid()) not in [12,14,16] and x.pt()>1e-5
)

# configure the papas fast simulation with the CMS detector
# help(Papas) for more information
# history nodes keeps track of which particles produced which tracks, clusters 
from PhysicsTools.HeppyCore.analyzers.PapasSim import PapasSim
# from PhysicsTools.HeppyCore.analyzers.Papas import Papas
from PhysicsTools.HeppyCore.papas.detectors.CMS import CMS
detector = CMS()

papas = cfg.Analyzer(
    PapasSim,
    instance_label = 'papas',
    detector = detector,
    gen_particles = 'gen_particles_stable',
    sim_particles = 'sim_particles',
    merged_ecals = 'ecal_clusters',
    merged_hcals = 'hcal_clusters',
    tracks = 'tracks', 
    output_history = 'history_nodes', 
    display_filter_func = lambda ptc: ptc.e()>1.,
    display = False,
    verbose = True
)


# group the clusters, tracks from simulation into connected blocks ready for reconstruction
from PhysicsTools.HeppyCore.analyzers.PapasPFBlockBuilder import PapasPFBlockBuilder
pfblocks = cfg.Analyzer(
    PapasPFBlockBuilder,
    tracks = 'tracks', 
    ecals = 'ecal_clusters', 
    hcals = 'hcal_clusters', 
    history = 'history_nodes',  
    output_blocks = 'reconstruction_blocks'
)


#reconstruct particles from blocks
from PhysicsTools.HeppyCore.analyzers.PapasPFReconstructor import PapasPFReconstructor
pfreconstruct = cfg.Analyzer(
    PapasPFReconstructor,
    instance_label = 'papas_PFreconstruction', 
    detector = detector,
    input_blocks = 'reconstruction_blocks',
    history = 'history_nodes',     
    output_particles_dict = 'particles_dict', 
    output_particles_list = 'particles_list'
)



# Use a Filter to select leptons from the output of papas simulation.
# Currently, we're treating electrons and muons transparently.
# we could use two different instances for the Filter module
# to get separate collections of electrons and muons
# help(Filter) for more information
from PhysicsTools.HeppyCore.analyzers.Filter import Filter
sim_electrons = cfg.Analyzer(
    Filter,
    'sim_electrons',
    output = 'sim_electrons',
    input_objects = 'papas_sim_particles',
    filter_func = lambda ptc: abs(ptc.pdgid()) in [11]
)

sim_muons = cfg.Analyzer(
    Filter,
    'sim_muons',
    output = 'sim_muons',
    input_objects = 'papas_sim_particles',
    filter_func = lambda ptc: abs(ptc.pdgid()) in [13]
)


# Applying a simple resolution and efficiency model to electrons and muons.
# Indeed, papas simply copies generated electrons and muons
# from its input gen particle collection to its output reconstructed
# particle collection.
# Setting up the electron and muon models is left to the user,
# and the LeptonSmearer is just an example
# help(LeptonSmearer) for more information
from PhysicsTools.HeppyCore.analyzers.GaussianSmearer import GaussianSmearer     
def accept_electron(ele):
    return abs(ele.eta()) < 2.5 and ele.e() > 5.
electrons = cfg.Analyzer(
    GaussianSmearer,
    'electrons',
    output = 'electrons',
    input_objects = 'sim_electrons',
    accept=accept_electron, 
    mu_sigma=(1, 0.1)
    )

def accept_muon(mu):
    return abs(mu.eta()) < 2.5 and mu.pt() > 5.
muons = cfg.Analyzer(
    GaussianSmearer,
    'muons',
    output = 'muons',
    input_objects = 'sim_muons',
    accept=accept_muon, 
    mu_sigma=(1, 0.02)
    )


#merge smeared leptons with the reconstructed particles
from PhysicsTools.HeppyCore.analyzers.Merger import Merger
from PhysicsTools.HeppyCore.particles.p4 import P4
merge_particles = cfg.Analyzer(
    Merger,
    instance_label = 'merge_particles',
    inputs=['papas_PFreconstruction_particles_list', 'electrons', 'muons'], 
    output = 'rec_particles',
    sort_key = P4.sort_key
)

papas_sequence = [
    gen_particles_stable,
    papas,
    pfblocks,
    pfreconstruct,
    sim_electrons,
    sim_muons, 
    electrons,
    muons, 
#    select_leptons,
#    smear_leptons,
    merge_particles, 
]
