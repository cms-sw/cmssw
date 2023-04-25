/* 
 *  \class EcalMatacqAnalyzer
 *
 *  primary author: Gautier Hamel De Monchenault - CEA/Saclay
 *  author: Julie Malcles - CEA/Saclay
 */

#include <TFile.h>
#include <TTree.h>
#include <TProfile.h>
#include <TChain.h>
#include <vector>

#include <CalibCalorimetry/EcalLaserAnalyzer/plugins/EcalMatacqAnalyzer.h>

#include <sstream>
#include <iostream>
#include <iomanip>

#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <FWCore/Utilities/interface/Exception.h>

#include <DataFormats/EcalDetId/interface/EcalDetIdCollections.h>

#include <FWCore/Framework/interface/Event.h>
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMatacq.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMom.h>
#include <CalibCalorimetry/EcalLaserAnalyzer/interface/TMTQ.h>

//========================================================================
EcalMatacqAnalyzer::EcalMatacqAnalyzer(const edm::ParameterSet& iConfig)
    //========================================================================
    :

      iEvent(0),

      // framework parameters with default values

      _presample(iConfig.getUntrackedParameter<double>("nPresamples", 6.7)),
      _nsamplesaftmax(iConfig.getUntrackedParameter<unsigned int>("nSamplesAftMax", 80)),
      _noiseCut(iConfig.getUntrackedParameter<unsigned int>("noiseCut", 7)),
      _parabnbefmax(iConfig.getUntrackedParameter<unsigned int>("paraBeforeMax", 8)),
      _parabnaftmax(iConfig.getUntrackedParameter<unsigned int>("paraAfterMax", 7)),
      _thres(iConfig.getUntrackedParameter<unsigned int>("threshold", 10)),
      _lowlev(iConfig.getUntrackedParameter<unsigned int>("lowLevel", 20)),
      _highlev(iConfig.getUntrackedParameter<unsigned int>("highLevel", 80)),
      _nevlasers(iConfig.getUntrackedParameter<unsigned int>("nEventLaser", 600)),
      _timebefmax(iConfig.getUntrackedParameter<unsigned int>("timeBefMax", 100)),
      _timeaftmax(iConfig.getUntrackedParameter<unsigned int>("timeAftMax", 250)),
      _cutwindow(iConfig.getUntrackedParameter<double>("cutWindow", 0.1)),
      _nsamplesshape(iConfig.getUntrackedParameter<unsigned int>("nSamplesShape", 250)),
      _presampleshape(iConfig.getUntrackedParameter<unsigned int>("nPresamplesShape", 50)),
      _slide(iConfig.getUntrackedParameter<unsigned int>("nSlide", 100)),
      _fedid(iConfig.getUntrackedParameter<int>("fedID", -999)),
      _debug(iConfig.getUntrackedParameter<int>("debug", 0)),
      resdir_(iConfig.getUntrackedParameter<std::string>("resDir")),
      digiCollection_(iConfig.getParameter<std::string>("digiCollection")),
      digiProducer_(iConfig.getParameter<std::string>("digiProducer")),
      eventHeaderCollection_(iConfig.getParameter<std::string>("eventHeaderCollection")),
      eventHeaderProducer_(iConfig.getParameter<std::string>("eventHeaderProducer")),
      pmatToken_(consumes<EcalMatacqDigiCollection>(edm::InputTag(digiProducer_, digiCollection_))),
      dccToken_(consumes<EcalRawDataCollection>(edm::InputTag(eventHeaderProducer_, eventHeaderCollection_))),
      nSides(NSIDES),
      lightside(0),
      runType(-1),
      runNum(0),
      event(0),
      color(-1),
      maxsamp(0),
      nsamples(0),
      tt(0)

//========================================================================
{
  //now do what ever initialization is needed
}

//========================================================================
void EcalMatacqAnalyzer::beginJob() {
  //========================================================================

  // Define temporary file name

  sampfile = resdir_;
  sampfile += "/TmpTreeMatacqAnalyzer.root";

  sampFile = new TFile(sampfile.c_str(), "RECREATE");

  // declaration of the tree to fill

  tree = new TTree("MatacqTree", "MatacqTree");

  //List of branches

  tree->Branch("event", &event, "event/I");
  tree->Branch("color", &color, "color/I");
  tree->Branch("matacq", &matacq, "matacq[2560]/D");
  tree->Branch("nsamples", &nsamples, "nsamples/I");
  tree->Branch("maxsamp", &maxsamp, "maxsamp/I");
  tree->Branch("tt", &tt, "tt/D");
  tree->Branch("lightside", &lightside, "lightside/I");

  tree->SetBranchAddress("event", &event);
  tree->SetBranchAddress("color", &color);
  tree->SetBranchAddress("matacq", matacq);
  tree->SetBranchAddress("nsamples", &nsamples);
  tree->SetBranchAddress("maxsamp", &maxsamp);
  tree->SetBranchAddress("tt", &tt);
  tree->SetBranchAddress("lightside", &lightside);

  // Define output results files' names

  std::stringstream namefile;
  namefile << resdir_ << "/MATACQ.root";
  outfile = namefile.str();

  // Laser events counter
  laserEvents = 0;
  isThereMatacq = false;
}

//========================================================================
void EcalMatacqAnalyzer::analyze(const edm::Event& e, const edm::EventSetup& c) {
  //========================================================================

  ++iEvent;

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- Entering Analyze -- event= " << iEvent;

  // retrieving MATACQ :
  const edm::Handle<EcalMatacqDigiCollection>& pmatacqDigi = e.getHandle(pmatToken_);
  const EcalMatacqDigiCollection* matacqDigi = (pmatacqDigi.isValid()) ? pmatacqDigi.product() : nullptr;
  if (pmatacqDigi.isValid()) {
    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- Matacq Digis Found -- ";
  } else {
    edm::LogError("EcalMatacqAnalyzzer") << "Error! can't get the product EcalMatacqDigi producer:"
                                         << digiProducer_.c_str() << " collection:" << digiCollection_.c_str();
    return;
  }

  // retrieving DCC header

  const edm::Handle<EcalRawDataCollection>& pDCCHeader = e.getHandle(dccToken_);
  const EcalRawDataCollection* DCCHeader = (pDCCHeader.isValid()) ? pDCCHeader.product() : nullptr;
  if (!pDCCHeader.isValid()) {
    edm::LogError("EcalMatacqAnalyzzer") << "Error! can't get the product EcalRawData producer:"
                                         << eventHeaderProducer_.c_str()
                                         << " collection:" << eventHeaderCollection_.c_str();
    return;
  }

  // ====================================
  // Decode Basic DCCHeader Information
  // ====================================

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- Before header -- ";

  for (EcalRawDataCollection::const_iterator headerItr = DCCHeader->begin(); headerItr != DCCHeader->end();
       ++headerItr) {
    EcalDCCHeaderBlock::EcalDCCEventSettings settings = headerItr->getEventSettings();
    color = (int)settings.wavelength;
    if (color < 0)
      return;

    // Get run type and run number

    int fed = headerItr->fedId();

    if (fed != _fedid && _fedid != -999)
      continue;

    runType = headerItr->getRunType();
    runNum = headerItr->getRunNumber();
    event = headerItr->getLV1();

    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- runtype:" << runType << " event:" << event << " runNum:" << runNum;

    dccID = headerItr->getDccInTCCCommand();
    fedID = headerItr->fedId();
    lightside = headerItr->getRtHalf();

    //assert (lightside<2 && lightside>=0);

    if (lightside != 1 && lightside != 0) {
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "Unexpected lightside: " << lightside << " for event " << iEvent;
      return;
    }
    if (_debug == 1) {
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- Inside header before fed cut -- color=" << color << ", dcc=" << dccID
          << ", fed=" << fedID << ",  lightside=" << lightside << ", runType=" << runType;
    }

    // take event only if the fed corresponds to the DCC in TCC
    if (600 + dccID != fedID)
      continue;

    if (_debug == 1) {
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- Inside header after fed cut -- color=" << color << ", dcc=" << dccID << ", fed=" << fedID
          << ",  lightside=" << lightside << ", runType=" << runType;
    }

    // Cut on runType

    if (runType != EcalDCCHeaderBlock::LASER_STD && runType != EcalDCCHeaderBlock::LASER_GAP &&
        runType != EcalDCCHeaderBlock::LASER_POWER_SCAN && runType != EcalDCCHeaderBlock::LASER_DELAY_SCAN)
      return;

    std::vector<int>::iterator iter = find(colors.begin(), colors.end(), color);
    if (iter == colors.end()) {
      colors.push_back(color);
      edm::LogVerbatim("EcalMatacqAnalyzzer") << " new color found " << color << " " << colors.size();
    }
  }

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- Before digis -- Event:" << iEvent;

  // Count laser events
  laserEvents++;

  // ===========================
  // Decode Matacq Information
  // ===========================

  int iCh = 0;
  double max = 0;

  for (EcalMatacqDigiCollection::const_iterator it = matacqDigi->begin(); it != matacqDigi->end();
       ++it) {  // Loop on matacq channel

    //
    const EcalMatacqDigi& digis = *it;

    //if(digis.size()==0 || iCh>=N_channels) continue;
    if (_debug == 1) {
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- Inside digis -- digi size=" << digis.size();
    }

    if (digis.size() == 0)
      continue;
    else
      isThereMatacq = true;

    max = 0;
    maxsamp = 0;
    nsamples = digis.size();
    tt = digis.tTrig();

    for (int i = 0; i < digis.size(); ++i) {  // Loop on matacq samples
      matacq[i] = -digis.adcCount(i);
      if (matacq[i] > max) {
        max = matacq[i];
        maxsamp = i;
      }
    }
    if (_debug == 1) {
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- Inside digis -- nsamples=" << nsamples << ", max=" << max;
    }

    iCh++;
  }

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- After digis -- Event: " << iEvent;
  tree->Fill();

}  // analyze

//========================================================================
void EcalMatacqAnalyzer::endJob() {
  // Don't do anything if there is no events
  if (!isThereMatacq) {
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "\t+=+     WARNING! NO MATACQ        +=+";
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";

    // Remove temporary file
    FILE* test;
    test = fopen(sampfile.c_str(), "r");
    if (test) {
      std::stringstream del2;
      del2 << "rm " << sampfile;
      system(del2.str().c_str());
    }
    return;
  }

  assert(colors.size() <= nColor);
  unsigned int nCol = colors.size();

  for (unsigned int iCol = 0; iCol < nCol; iCol++) {
    for (unsigned int iSide = 0; iSide < nSide; iSide++) {
      MTQ[iCol][iSide] = new TMTQ();
    }
  }

  outFile = new TFile(outfile.c_str(), "RECREATE");

  TProfile* shapeMat = new TProfile("shapeLaser", "shapeLaser", _nsamplesshape, -0.5, double(_nsamplesshape) - 0.5);
  TProfile* shapeMatTmp = new TProfile(
      "shapeLaserTmp", "shapeLaserTmp", _timeaftmax + _timebefmax, -0.5, double(_timeaftmax + _timebefmax) - 0.5);

  edm::LogVerbatim("EcalMatacqAnalyzzer") << "\n\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
  edm::LogVerbatim("EcalMatacqAnalyzzer") << "\t+=+     Analyzing MATACQ data     +=+";

  //
  // create output ntuple
  //

  mtqShape = new TTree("MatacqShape", "MatacqShape");

  // list of branches
  // keep Patrice's notations

  mtqShape->Branch("event", &event, "event/I");
  mtqShape->Branch("color", &color, "color/I");
  mtqShape->Branch("status", &status, "status/I");
  mtqShape->Branch("peak", &peak, "peak/D");
  mtqShape->Branch("sigma", &sigma, "sigma/D");
  mtqShape->Branch("fit", &fit, "fit/D");
  mtqShape->Branch("ampl", &ampl, "ampl/D");
  mtqShape->Branch("trise", &trise, "trise/D");
  mtqShape->Branch("fwhm", &fwhm, "fwhm/D");
  mtqShape->Branch("fw20", &fw20, "fw20/D");
  mtqShape->Branch("fw80", &fw80, "fw80/D");
  mtqShape->Branch("ped", &ped, "ped/D");
  mtqShape->Branch("pedsig", &pedsig, "pedsig/D");
  mtqShape->Branch("ttrig", &ttrig, "ttrig/D");
  mtqShape->Branch("sliding", &sliding, "sliding/D");

  mtqShape->SetBranchAddress("event", &event);
  mtqShape->SetBranchAddress("color", &color);
  mtqShape->SetBranchAddress("status", &status);
  mtqShape->SetBranchAddress("peak", &peak);
  mtqShape->SetBranchAddress("sigma", &sigma);
  mtqShape->SetBranchAddress("fit", &fit);
  mtqShape->SetBranchAddress("ampl", &ampl);
  mtqShape->SetBranchAddress("fwhm", &fwhm);
  mtqShape->SetBranchAddress("fw20", &fw20);
  mtqShape->SetBranchAddress("fw80", &fw80);
  mtqShape->SetBranchAddress("trise", &trise);
  mtqShape->SetBranchAddress("ped", &ped);
  mtqShape->SetBranchAddress("pedsig", &pedsig);
  mtqShape->SetBranchAddress("ttrig", &ttrig);
  mtqShape->SetBranchAddress("sliding", &sliding);

  unsigned int endsample;
  unsigned int presample;

  // loop over the entries of the tree
  TChain* fChain = (TChain*)tree;
  Long64_t nentries = fChain->GetEntriesFast();

  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    // load the event
    Long64_t ientry = fChain->LoadTree(jentry);
    if (ientry < 0)
      break;
    fChain->GetEntry(jentry);

    status = 0;
    peak = -1;
    sigma = 0;
    fit = -1;
    ampl = -1;
    trise = -1;
    ttrig = tt;
    fwhm = 0;
    fw20 = 0;
    fw80 = 0;
    ped = 0;
    pedsig = 0;
    sliding = 0;

    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- inside loop 1  -- jentry:" << jentry << " over nentries=" << nentries;

    // create the object for Matacq data analysis

    endsample = maxsamp + _nsamplesaftmax;
    presample = int(_presample * nsamples / 100.);
    TMatacq* mtq = new TMatacq(nsamples,
                               presample,
                               endsample,
                               _noiseCut,
                               _parabnbefmax,
                               _parabnaftmax,
                               _thres,
                               _lowlev,
                               _highlev,
                               _nevlasers,
                               _slide);

    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 2  -- ";

    // analyse the Matacq data
    if (mtq->rawPulseAnalysis(nsamples, &matacq[0]) == 0) {
      status = 1;
      ped = mtq->getBaseLine();
      pedsig = mtq->getsigBaseLine();

      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 3  -- ped:" << ped;
      if (mtq->findPeak() == 0) {
        peak = mtq->getTimpeak();
        sigma = mtq->getsigTimpeak();
      }
      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 4  -- peak:" << peak;
      if (mtq->doFit() == 0) {
        fit = mtq->getTimax();
        ampl = mtq->getAmpl();
        fwhm = mtq->getFwhm();
        fw20 = mtq->getWidth20();
        fw80 = mtq->getWidth80();
        sliding = mtq->getSlide();
      }
      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 4  -- ampl:" << ampl;
      if (mtq->compute_trise() == 0) {
        trise = mtq->getTrise();
      }
      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 5  -- trise:" << trise;
    }

    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 6  -- status:" << status;

    if (status == 1 && mtq->findPeak() == 0) {
      int firstS = int(peak - double(_timebefmax));
      int lastS = int(peak + double(_timeaftmax));

      // Fill histo if there are enough samples
      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer")
            << "-- debug test -- inside loop 7  -- firstS:" << firstS << ", nsamples:" << nsamples;

      if (firstS >= 0 && lastS <= nsamples) {
        for (int i = firstS; i < lastS; i++) {
          shapeMatTmp->Fill(double(i) - firstS, matacq[i]);
        }

      } else {  // else extrapolate

        int firstSBis;

        if (firstS < 0) {  // fill first bins with 0

          double thisped;
          thisped = (matacq[0] + matacq[1] + matacq[2] + matacq[4] + matacq[5]) / 5.0;

          for (int i = firstS; i < 0; i++) {
            shapeMatTmp->Fill(double(i) - firstS, thisped);
          }
          firstSBis = 0;

        } else {
          firstSBis = firstS;
        }

        if (lastS > nsamples) {
          for (int i = firstSBis; i < int(nsamples); i++) {
            shapeMatTmp->Fill(double(i) - firstS, matacq[i]);
          }

          //extrapolate with expo tail

          double expb = 0.998;
          double matacqval = expb * matacq[nsamples - 1];

          for (int i = nsamples; i < lastS; i++) {
            shapeMatTmp->Fill(double(i) - firstS, matacqval);
            matacqval *= expb;
          }

        } else {
          for (int i = firstSBis; i < lastS; i++) {
            shapeMatTmp->Fill(double(i) - firstS, matacq[i]);
          }
        }
      }
    }
    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 8";

    // get back color

    int iCol = nCol;
    for (unsigned int i = 0; i < nCol; i++) {
      if (color == colors[i]) {
        iCol = i;
        i = nCol;
      }
    }
    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer")
          << "-- debug test -- inside loop 8bis color:" << color << " iCol:" << iCol << " nCol:" << nCol;

    // fill TMTQ

    if (status == 1 && mtq->findPeak() == 0 && mtq->doFit() == 0 && mtq->compute_trise() == 0)
      MTQ[iCol][lightside]->addEntry(peak, sigma, fit, ampl, trise, fwhm, fw20, fw80, ped, pedsig, sliding);

    // fill the output tree

    if (_debug == 1)
      edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside loop 9";
    mtqShape->Fill();

    // clean up
    delete mtq;
  }

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- after loop ";
  sampFile->Close();

  double Peak[6], Sigma[6], Fit[6], Ampl[6], Trise[6], Fwhm[6], Fw20[6], Fw80[6], Ped[6], Pedsig[6], Sliding[6];
  int Side;

  for (unsigned int iColor = 0; iColor < nCol; iColor++) {
    std::stringstream nametree;
    nametree << "MatacqCol" << colors[iColor];
    meanTree[iColor] = new TTree(nametree.str().c_str(), nametree.str().c_str());
    meanTree[iColor]->Branch("side", &Side, "Side/I");
    meanTree[iColor]->Branch("peak", &Peak, "Peak[6]/D");
    meanTree[iColor]->Branch("sigma", &Sigma, "Sigma[6]/D");
    meanTree[iColor]->Branch("fit", &Fit, "Fit[6]/D");
    meanTree[iColor]->Branch("ampl", &Ampl, "Ampl[6]/D");
    meanTree[iColor]->Branch("trise", &Trise, "Trise[6]/D");
    meanTree[iColor]->Branch("fwhm", &Fwhm, "Fwhm[6]/D");
    meanTree[iColor]->Branch("fw20", &Fw20, "Fw20[6]/D");
    meanTree[iColor]->Branch("fw80", &Fw80, "Fw80[6]/D");
    meanTree[iColor]->Branch("ped", &Ped, "Ped[6]/D");
    meanTree[iColor]->Branch("pedsig", &Pedsig, "Pedsig[6]/D");
    meanTree[iColor]->Branch("sliding", &Sliding, "Sliding[6]/D");

    meanTree[iColor]->SetBranchAddress("side", &Side);
    meanTree[iColor]->SetBranchAddress("peak", Peak);
    meanTree[iColor]->SetBranchAddress("sigma", Sigma);
    meanTree[iColor]->SetBranchAddress("fit", Fit);
    meanTree[iColor]->SetBranchAddress("ampl", Ampl);
    meanTree[iColor]->SetBranchAddress("fwhm", Fwhm);
    meanTree[iColor]->SetBranchAddress("fw20", Fw20);
    meanTree[iColor]->SetBranchAddress("fw80", Fw80);
    meanTree[iColor]->SetBranchAddress("trise", Trise);
    meanTree[iColor]->SetBranchAddress("ped", Ped);
    meanTree[iColor]->SetBranchAddress("pedsig", Pedsig);
    meanTree[iColor]->SetBranchAddress("sliding", Sliding);
  }

  for (unsigned int iCol = 0; iCol < nCol; iCol++) {
    for (unsigned int iSide = 0; iSide < nSides; iSide++) {
      Side = iSide;
      std::vector<double> val[TMTQ::nOutVar];

      for (int iVar = 0; iVar < TMTQ::nOutVar; iVar++) {
        val[iVar] = MTQ[iCol][iSide]->get(iVar);

        for (unsigned int i = 0; i < val[iVar].size(); i++) {
          switch (iVar) {
            case TMTQ::iPeak:
              Peak[i] = val[iVar].at(i);
              break;
            case TMTQ::iSigma:
              Sigma[i] = val[iVar].at(i);
              break;
            case TMTQ::iFit:
              Fit[i] = val[iVar].at(i);
              break;
            case TMTQ::iAmpl:
              Ampl[i] = val[iVar].at(i);
              break;
            case TMTQ::iFwhm:
              Fwhm[i] = val[iVar].at(i);
              break;
            case TMTQ::iFw20:
              Fw20[i] = val[iVar].at(i);
              break;
            case TMTQ::iFw80:
              Fw80[i] = val[iVar].at(i);
              break;
            case TMTQ::iTrise:
              Trise[i] = val[iVar].at(i);
              break;
            case TMTQ::iPed:
              Ped[i] = val[iVar].at(i);
              break;
            case TMTQ::iPedsig:
              Pedsig[i] = val[iVar].at(i);
              break;
            case TMTQ::iSlide:
              Sliding[i] = val[iVar].at(i);
              break;
          }
        }
      }
      meanTree[iCol]->Fill();
      if (_debug == 1)
        edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- inside final loop  ";
    }
  }

  // Calculate maximum with pol 2

  int im = shapeMatTmp->GetMaximumBin();
  double q1 = shapeMatTmp->GetBinContent(im - 1);
  double q2 = shapeMatTmp->GetBinContent(im);
  double q3 = shapeMatTmp->GetBinContent(im + 1);

  double a2 = (q3 + q1) / 2.0 - q2;
  double a1 = q2 - q1 + a2 * (1 - 2 * im);
  double a0 = q2 - a1 * im - a2 * im * im;

  double am = a0 - a1 * a1 / (4 * a2);

  // Compute pedestal

  double bl = 0;
  for (unsigned int i = 1; i < _presampleshape + 1; i++) {
    bl += shapeMatTmp->GetBinContent(i);
  }
  bl /= _presampleshape;

  // Compute and save laser shape

  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- computing shape  ";

  int firstBin = 0;
  double height = 0.0;

  for (unsigned int i = _timebefmax; i > _presampleshape; i--) {
    height = shapeMatTmp->GetBinContent(i) - bl;

    if (height < (am - bl) * _cutwindow) {
      firstBin = i;
      i = _presampleshape;
    }
  }

  unsigned int lastBin = firstBin + _nsamplesshape;

  for (unsigned int i = firstBin; i < lastBin; i++) {
    shapeMat->Fill(i - firstBin, shapeMatTmp->GetBinContent(i) - bl);
  }

  mtqShape->Write();
  for (unsigned int iColor = 0; iColor < nCol; iColor++) {
    meanTree[iColor]->Write();
  }
  if (_debug == 1)
    edm::LogVerbatim("EcalMatacqAnalyzzer") << "-- debug test -- writing  ";
  shapeMat->Write();

  // close the output file
  outFile->Close();

  // Remove temporary file
  FILE* test;
  test = fopen(sampfile.c_str(), "r");
  if (test) {
    std::stringstream del2;
    del2 << "rm " << sampfile;
    system(del2.str().c_str());
  }

  edm::LogVerbatim("EcalMatacqAnalyzzer") << "\t+=+    .................... done  +=+";
  edm::LogVerbatim("EcalMatacqAnalyzzer") << "\t+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+";
}

DEFINE_FWK_MODULE(EcalMatacqAnalyzer);
