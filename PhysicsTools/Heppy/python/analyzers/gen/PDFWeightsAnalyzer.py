from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
import PhysicsTools.HeppyCore.framework.config as cfg

        
class PDFWeightsAnalyzer( Analyzer ):
    """    """
    def __init__(self, cfg_ana, cfg_comp, looperName ):
        super(PDFWeightsAnalyzer,self).__init__(cfg_ana,cfg_comp,looperName)
        self.doPDFWeights = hasattr(self.cfg_ana, "PDFWeights") and len(self.cfg_ana.PDFWeights) > 0
        self.doPDFVars = hasattr(self.cfg_ana, "doPDFVars") and self.cfg_ana.doPDFVars == True
        if self.doPDFWeights:
            self.pdfWeightInit = False
    #---------------------------------------------
    # DECLARATION OF HANDLES OF GEN LEVEL OBJECTS 
    #---------------------------------------------
        

    def declareHandles(self):
        super(PDFWeightsAnalyzer, self).declareHandles()

        if self.doPDFVars or self.doPDFWeights:
            self.mchandles['pdfstuff'] = AutoHandle( 'generator', 'GenEventInfoProduct' )

    def beginLoop(self, setup):
        super(PDFWeightsAnalyzer,self).beginLoop(setup)

    def initPDFWeights(self):
        from ROOT import PdfWeightProducerTool
        self.pdfWeightInit = True
        self.pdfWeightTool = PdfWeightProducerTool()
        for pdf in self.cfg_ana.PDFWeights:
            self.pdfWeightTool.addPdfSet(pdf+".LHgrid")
        self.pdfWeightTool.beginJob()

    def makePDFWeights(self, event):
        if not self.pdfWeightInit: self.initPDFWeights()
        self.pdfWeightTool.processEvent(self.genInfo)
        event.pdfWeights = {}
        for pdf in self.cfg_ana.PDFWeights:
            ws = self.pdfWeightTool.getWeights(pdf+".LHgrid")
            event.pdfWeights[pdf] = [w for w in ws]

    def process(self, event):
        self.readCollections( event.input )

        # if not MC, nothing to do
        if not self.cfg_comp.isMC: 
            return True

        if self.doPDFVars or self.doPDFWeights:
            self.genInfo = self.mchandles['pdfstuff'].product()
        if self.doPDFWeights:
            self.makePDFWeights(event)
        if self.doPDFVars:
            event.pdf_x1 = self.genInfo.pdf().x.first
            event.pdf_x2 = self.genInfo.pdf().x.second
            event.pdf_id1 = self.genInfo.pdf().id.first
            event.pdf_id2 = self.genInfo.pdf().id.second
            event.pdf_scale = self.genInfo.pdf().scalePDF

        return True

setattr(PDFWeightsAnalyzer,"defaultConfig",
    cfg.Analyzer(PDFWeightsAnalyzer,
        PDFWeights = []
    )
)
