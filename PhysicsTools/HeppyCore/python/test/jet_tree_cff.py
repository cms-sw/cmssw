import PhysicsTools.HeppyCore.framework.config as cfg

def jet_tree_sequence(gen_ptcs, rec_ptcs, njets, ptmin):

    fastjet_args = None
    if njets:
        fastjet_args = dict(njets=njets)
    else:
        fastjet_args = dict(ptmin=ptmin)
    
    
    from PhysicsTools.HeppyCore.analyzers.fcc.JetClusterizer import JetClusterizer
    gen_jets = cfg.Analyzer(
        JetClusterizer,
        output = 'gen_jets',
        particles = gen_ptcs,
        fastjet_args = fastjet_args,
        )

    jets = cfg.Analyzer(
        JetClusterizer,
        output = 'jets', 
        particles = rec_ptcs,
        fastjet_args = fastjet_args,
        )

    from PhysicsTools.HeppyCore.analyzers.Matcher import Matcher
    jet_match = cfg.Analyzer(
        Matcher,
        match_particles = 'jets',
        particles = 'gen_jets',
        delta_r = 0.3
        )

    from PhysicsTools.HeppyCore.analyzers.JetTreeProducer import JetTreeProducer
    jet_tree = cfg.Analyzer(
        JetTreeProducer,
        tree_name = 'events',
        tree_title = 'jets',
        jets = 'gen_jets'
        )

    return [gen_jets, jets, jet_match, jet_tree]
