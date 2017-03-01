import ROOT
import numpy as np

class BTagWeightCalculator:
    """
    Calculates the jet and event correction factor as a weight based on the b-tagger shape-dependent data/mc 
    corrections.

    Currently, the recipe is only described in https://twiki.cern.ch/twiki/bin/viewauth/CMS/TTbarHbbRun2ReferenceAnalysis#Applying_CSV_weights

    In short, jet-by-jet correction factors as a function of pt, eta and CSV have been derived.
    This code accesses the flavour of MC jets and gets the correct weight histogram
    corresponding to the pt, eta and flavour of the jet.
    From there, the per-jet weight is just accessed according to the value of the discriminator.
    """
    def __init__(self, fn_hf, fn_lf) :
        self.pdfs = {}

        #bin edges of the heavy-flavour histograms
        #pt>=20 && pt<30 -> bin=0
        #pt>=30 && pt<40 -> bin=1
        #etc
        self.pt_bins_hf = np.array([20, 30, 40, 60, 100])
        self.eta_bins_hf = np.array([0, 2.41])

        #bin edges of the light-flavour histograms 
        self.pt_bins_lf = np.array([20, 30, 40, 60])
        self.eta_bins_lf = np.array([0, 0.8, 1.6, 2.41])

        #name of the default b-tagger
        self.btag = "pfCombinedInclusiveSecondaryVertexV2BJetTags"
        self.init(fn_hf, fn_lf)

        # systematic uncertainties for different flavour assignments
        self.systematics_for_b = ["JESUp", "JESDown", "LFUp", "LFDown",
                                  "HFStats1Up", "HFStats1Down", "HFStats2Up", "HFStats2Down"]
        self.systematics_for_c = ["cErr1Up", "cErr1Down", "cErr2Up", "cErr2Down"]
        self.systematics_for_l = ["JESUp", "JESDown", "HFUp", "HFDown",
                                  "LFStats1Up", "LFStats1Down", "LFStats2Up", "LFStats2Down"]

    def getBin(self, bvec, val):
        return int(bvec.searchsorted(val, side="right")) - 1

    def init(self, fn_hf, fn_lf):
        """
        fn_hf (string) - path to the heavy flavour weight file
        fn_lf (string) - path to the light flavour weight file
        """
        print "[BTagWeightCalculator]: Initializing from files", fn_hf, fn_lf

        #print "hf"
        self.pdfs["hf"] = self.getHistosFromFile(fn_hf)
        #print "lf"
        self.pdfs["lf"] = self.getHistosFromFile(fn_lf)

        return True

    def getHistosFromFile(self, fn):
        """
        Initialized the lookup table for b-tag weight histograms based on jet
        pt, eta and flavour.
        The format of the weight file is similar to:
         KEY: TH1D     csv_ratio_Pt0_Eta0_final;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_JESUp;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_JESDown;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_LFUp;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_LFDown;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_Stats1Up;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_Stats1Down;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_Stats2Up;1
         KEY: TH1D     csv_ratio_Pt0_Eta0_final_Stats2Down;1
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final;2
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final;1
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final_cErr1Up;1
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final_cErr1Down;1
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final_cErr2Up;1
         KEY: TH1D     c_csv_ratio_Pt0_Eta0_final_cErr2Down;1
        """
        ret = {}
        tf = ROOT.TFile(fn)
        if not tf or tf.IsZombie():
            raise FileNotFoundError("Could not open file {0}".format(fn))
        ROOT.gROOT.cd()
        for k in tf.GetListOfKeys():
            kn = k.GetName()
            if not (kn.startswith("csv_ratio") or kn.startswith("c_csv_ratio") ):
                continue
            spl = kn.split("_")
            is_c = 1 if kn.startswith("c_csv_ratio") else 0

            if spl[2+is_c] == "all":
                ptbin = -1
                etabin = -1
                kind = "all"
                syst = "nominal"
            else:
                ptbin = int(spl[2+is_c][2:])
                etabin = int(spl[3+is_c][3:])
                kind = spl[4+is_c]
                if len(spl)==(6+is_c):
                    syst = spl[5+is_c]
                else:
                    syst = "nominal"
            ret[(is_c, ptbin, etabin, kind, syst)] = k.ReadObj().Clone()
        return ret

    def calcJetWeight(self, jet, kind, systematic):
        """
        Calculates the per-jet correction factor.
        jet: either an object with the attributes pt, eta, mcFlavour, self.btag
             or a Heppy Jet
        kind: string specifying the name of the corrections. Usually "final".
        systematic: the correction systematic, e.g. "nominal", "JESUp", etc
     """
        #if jet is a simple class with attributes
        if isinstance(getattr(jet, "pt"), float):
            pt   = getattr(jet, "pt")
            aeta = abs(getattr(jet, "eta"))
            fl   = abs(getattr(jet, "hadronFlavour"))
            csv  = getattr(jet, self.btag)
        #if jet is a heppy Jet object
        else:
            #print "could not get jet", e
            pt   = jet.pt()
            aeta = abs(jet.eta())
            fl   = abs(jet.hadronFlavour())
            csv  = jet.btag(self.btag)
        return self.calcJetWeightImpl(pt, aeta, fl, csv, kind, systematic)

    def calcJetWeightImpl(self, pt, aeta, fl, csv, kind, systematic):

        is_b = (fl == 5)
        is_c = (fl == 4)
        is_l = not (is_b or is_c)

        #if evaluating a weight for systematic uncertainties, make sure the jet is affected. If not, return 'nominal' weight
        if systematic != "nominal":
            if (is_b and systematic not in self.systematics_for_b) or (is_c and systematic not in self.systematics_for_c) or (is_l and systematic not in self.systematics_for_l):
                systematic = "nominal"

        #needed because the TH1 names for Stats are same for HF and LF
        if "Stats" in systematic:
            systematic = systematic[2:]

        if is_b or is_c:
            ptbin = self.getBin(self.pt_bins_hf, pt)
            etabin = self.getBin(self.eta_bins_hf, aeta)
        else:
            ptbin = self.getBin(self.pt_bins_lf, pt)
            etabin = self.getBin(self.eta_bins_lf, aeta)

        if ptbin < 0 or etabin < 0:
            #print "pt or eta bin outside range", pt, aeta, ptbin, etabin
            return 1.0

        k = (is_c, ptbin, etabin, kind, systematic)
        hdict = self.pdfs["lf"]
        if is_b or is_c:
            hdict = self.pdfs["hf"]
        h = hdict.get(k, None)
        if not h:
            #print "no histogram", k
            return 1.0

        if csv > 1:
            csv = 1
            
        csvbin = 1
        csvbin = h.FindBin(csv)
        #This is to fix csv=-10 not being accounted for in CSV SF input hists
        if csvbin <= 0:
            csvbin = 1
        if csvbin > h.GetNbinsX():
            csvbin = h.GetNbinsX()

        w = h.GetBinContent(csvbin)
        return w

    def calcEventWeight(self, jets, kind, systematic):
        """
        The per-event weight is just a product of per-jet weights.
        """
        weights = np.array(
            [self.calcJetWeight(jet, kind, systematic)
            for jet in jets]
        )

        wtot = np.prod(weights)
        return wtot
