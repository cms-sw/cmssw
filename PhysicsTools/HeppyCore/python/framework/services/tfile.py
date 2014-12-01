from PhysicsTools.HeppyCore.framework.services.service import Service
from ROOT import TFile

class TFileService(Service):
    '''TFile service.
    The file attribute is a TFile that can be used in several analyzers.
    The file is closed when the service stops.

    Example configuration:

    output_rootfile = cfg.Service(
      TFileService,
      'myhists',
      fname='histograms.root',
      option='recreate'
    )

    '''
    def __init__(self, cfg, comp, outdir):
        fname = '/'.join([outdir, cfg.fname])
        self.file = TFile(fname, cfg.option)
        
    def stop(self):
        self.file.Write() 
        self.file.Close()

if __name__ == '__main__':
    fileservice = TFileService('test.root', 'recreate')
    fileservice.start()
    fileservice.stop()
