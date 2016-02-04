/*
  This macro takes as input an histogram file produced by estimatePileup.py
  and gives as output a cff file to be copied to FastSimulation/PileUpProducer/python/
*/

void create_cff_from_histo(TString input="pudist.root",TString output="MyPileup_cff.py") {

  TFile* f = new TFile(input);
  TH1F* h = (TH1F*)f->Get("pileup");

  ofstream out(output);

  out << "from FastSimulation.Configuration.RandomServiceInitialization_cff import *" << endl;
  out << "from FastSimulation.PileUpProducer.PileUpSimulator7TeV_cfi import *" << endl;
  out << "from FastSimulation.Configuration.FamosSequences_cff import famosPileUp" << endl;
  out << "famosPileUp.PileUpSimulator = PileUpSimulatorBlock.PileUpSimulator" << endl;
  out << "famosPileUp.PileUpSimulator.usePoisson = False" << endl;

  out << "famosPileUp.PileUpSimulator.probFunctionVariable = (";
  for (int i=0; i<h->GetNbinsX()-1; i++) {
    if (i>0) out << ",";
    out << i;
  }
  out << ")" << endl;

  out << "famosPileUp.PileUpSimulator.probValue = (";
  for (int i=1; i<h->GetNbinsX(); i++) {//note: bin 0 is the underflow
    if (i>1) out << ",";
    out << h->GetBinContent(i)/h->GetEntries();
  }
  out << ")" << endl;

  cout << "Saved file " << output << endl;
  cout << "Copy it into FastSimulation/PileUpProducer/python/ to be able to load it from your configuration" << endl;

}
