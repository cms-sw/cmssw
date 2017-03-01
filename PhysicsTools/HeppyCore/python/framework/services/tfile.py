from PhysicsTools.HeppyCore.framework.services.service import Service
from ROOT import TFile

class TFileService(Service):
    """TFile service.
    The file attribute is a TFile that can be used in several analyzers.
    The file is closed when the service stops.

    Example configuration:

    output_rootfile = cfg.Service(
      TFileService,
      'myhists',
      fname='histograms.root',
      option='recreate'
    )
    """
    def __init__(self, cfg, comp, outdir):
        """
        cfg must contain:
        - fname: file name 
        - option: TFile options, e.g. recreate
        
        outdir is the output directory for the TFile

        comp is a dummy parameter here.  
        It is needed because the looper creates services and analyzers 
        in the same way, providing the configuration (cfg), 
        the component currently processed (comp), 
        and the output directory. 

        Other implementations of the TFileService could 
        make use of the component information, eg. the component name. 
        """
        fname = '/'.join([outdir, cfg.fname])
        self.file = TFile(fname, cfg.option)
        
    def stop(self):
        self.file.Write() 
        self.file.Close()

