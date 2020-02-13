#include "L1Trigger/DTPhase2Trigger/interface/HoughGrouping.h"

using namespace std;
using namespace edm;
using namespace cms;


struct {
  Bool_t operator()(std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> a, std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> b) const {

    UShort_t sumhqa    = 0; UShort_t sumhqb    = 0;
    UShort_t sumlqa    = 0; UShort_t sumlqb    = 0;
    Double_t sumdista  = 0; Double_t sumdistb  = 0;

    for (UShort_t lay = 0; lay < 8; lay++) {
      sumhqa   += (UShort_t)get<1>(a)[lay]; sumhqb   += (UShort_t)get<1>(b)[lay];
      sumlqa   += (UShort_t)get<2>(a)[lay]; sumlqb   += (UShort_t)get<2>(b)[lay];
      sumdista += get<4>(a)[lay];           sumdistb += get<4>(b)[lay];
    }

    if      (get<0>(a) != get<0>(b)) return (get<0>(a) > get<0>(b)); // number of layers with hits
    else if (sumhqa    != sumhqb)    return (sumhqa    > sumhqb);    // number of hq hits
    else if (sumlqa    != sumlqb)    return (sumlqa    > sumlqb);    // number of lq hits
    else if (get<3>(a) != get<3>(b)) return (get<3>(a) < get<3>(b)); // abs. diff. between SL1 & SL3 hits
    else                             return (sumdista < sumdistb);   // abs. dist. to digis
  }
} HoughOrdering;


// ============================================================================
// Constructors and destructor
// ============================================================================
HoughGrouping::HoughGrouping(const ParameterSet& pset): MotherGrouping(pset) {
  // Obtention of parameters
  debug                     = pset.getUntrackedParameter<Bool_t>("debug");
  if (debug) cout <<"HoughGrouping: constructor" << endl;
  
  // HOUGH TRANSFORM CONFIGURATION
  angletan                  = pset.getUntrackedParameter<Double_t>("angletan");
  anglebinwidth             = pset.getUntrackedParameter<Double_t>("anglebinwidth");
  posbinwidth               = pset.getUntrackedParameter<Double_t>("posbinwidth");
  
  // MAXIMA SEARCH CONFIGURATION
  maxdeltaAngDeg            = pset.getUntrackedParameter<Double_t>("maxdeltaAngDeg");
  maxdeltaPos               = pset.getUntrackedParameter<Double_t>("maxdeltaPos");
  UpperNumber               = (UShort_t)pset.getUntrackedParameter<Int_t>("UpperNumber");
  LowerNumber               = (UShort_t)pset.getUntrackedParameter<Int_t>("LowerNumber");
  
  // HITS ASSOCIATION CONFIGURATION
  MaxDistanceToWire         = pset.getUntrackedParameter<Double_t>("MaxDistanceToWire");
  
  // CANDIDATE QUALITY CONFIGURATION
  minNLayerHits             = (UShort_t)pset.getUntrackedParameter<Int_t>("minNLayerHits");
  minSingleSLHitsMax        = (UShort_t)pset.getUntrackedParameter<Int_t>("minSingleSLHitsMax");
  minSingleSLHitsMin        = (UShort_t)pset.getUntrackedParameter<Int_t>("minSingleSLHitsMin");
  allowUncorrelatedPatterns = pset.getUntrackedParameter<Bool_t>("allowUncorrelatedPatterns");
  minUncorrelatedHits       = (UShort_t)pset.getUntrackedParameter<Int_t>("minUncorrelatedHits");
}


HoughGrouping::~HoughGrouping() {
  if (debug) cout <<"HoughGrouping: destructor" << endl;
}



// ============================================================================
// Main methods (initialise, run, finish)
// ============================================================================
void HoughGrouping::initialise(const edm::EventSetup& iEventSetup) {
  if (debug) cout << "HoughGrouping::initialise" << endl;
  
  ResetAttributes();
  
  maxrads       = TMath::PiOver2() - TMath::ATan(angletan);
  minangle      = anglebinwidth * TMath::TwoPi() / 360;
  halfanglebins = TMath::Nint(maxrads / minangle + 1);
  anglebins     = (UShort_t) 2 * halfanglebins;
  oneanglebin   = maxrads / halfanglebins;
  
  maxdeltaAng   = maxdeltaAngDeg * TMath::TwoPi() / 360;
  
  // Initialisation of anglemap. Posmap depends on the size of the chamber.
  Double_t phi = 0; anglemap = {};
  for (UShort_t ab = 0; ab < halfanglebins; ab++) {
    anglemap[ab]  = phi;
    phi           += oneanglebin;
  }
  
  phi = (TMath::Pi() - maxrads);
  for (UShort_t ab = halfanglebins; ab < anglebins; ab++) {
    anglemap[ab]  = phi;
    phi           += oneanglebin;
  }
  
  linespace = new UShort_t*[anglebins];
  
  if (debug) {
    cout << "\nHoughGrouping::ResetAttributes - Information from the initialisation of HoughGrouping:" << endl;
    cout << "HoughGrouping::ResetAttributes - maxrads: "       << maxrads       << endl;
    cout << "HoughGrouping::ResetAttributes - anglebinwidth: " << anglebinwidth << endl;
    cout << "HoughGrouping::ResetAttributes - minangle: "      << minangle      << endl;
    cout << "HoughGrouping::ResetAttributes - halfanglebins: " << halfanglebins << endl;
    cout << "HoughGrouping::ResetAttributes - anglebins: "     << anglebins     << endl;
    cout << "HoughGrouping::ResetAttributes - oneanglebin: "   << oneanglebin   << endl;
    cout << "HoughGrouping::ResetAttributes - posbinwidth: "   << posbinwidth   << endl;
  }
}


void HoughGrouping::run(edm::Event& iEvent, const edm::EventSetup& iEventSetup, DTDigiCollection digis, std::vector<MuonPath*> *outMpath) {
  if (debug) cout << "\nHoughGrouping::run" << endl;

  ResetAttributes();
  
  iEventSetup.get<MuonGeometryRecord>().get(dtGeomH);
  const DTGeometry* dtGeom = dtGeomH.product();
  
  if (debug) cout << "HoughGrouping::run - Beginning digis' loop..." << endl;
  LocalPoint wirePosInLay, wirePosInChamber; GlobalPoint wirePosGlob;
  for (DTDigiCollection::DigiRangeIterator dtLayerIdIt = digis.begin();           dtLayerIdIt != digis.end();              dtLayerIdIt++) {
    const DTLayer* lay   = dtGeom->layer((*dtLayerIdIt).first);
    for (DTDigiCollection::const_iterator  digiIt = ((*dtLayerIdIt).second).first; digiIt != ((*dtLayerIdIt).second).second; digiIt++) {
      if (debug) {
        cout << "\nHoughGrouping::run - Digi number " << idigi                             << endl;
        cout << "HoughGrouping::run - Wheel: "        << (*dtLayerIdIt).first.wheel()      << endl;
        cout << "HoughGrouping::run - Chamber: "      << (*dtLayerIdIt).first.station()    << endl;
        cout << "HoughGrouping::run - Sector: "       << (*dtLayerIdIt).first.sector()     << endl;
        cout << "HoughGrouping::run - Superlayer: "   << (*dtLayerIdIt).first.superLayer() << endl;
        cout << "HoughGrouping::run - Layer: "        << (*dtLayerIdIt).first.layer()      << endl;
        cout << "HoughGrouping::run - Wire: "         << (*digiIt).wire()                  << endl;
        cout << "HoughGrouping::run - First wire: "   << lay->specificTopology().firstChannel() << endl;
        cout << "HoughGrouping::run - Last wire: "    << lay->specificTopology().lastChannel()  << endl;
        cout << "HoughGrouping::run - First wire x: " << lay->specificTopology().wirePosition(lay->specificTopology().firstChannel()) << endl;
        cout << "HoughGrouping::run - Last wire x: "  << lay->specificTopology().wirePosition(lay->specificTopology().lastChannel())  << endl;
        cout << "HoughGrouping::run - Cell width: "   << lay->specificTopology().cellWidth()    << endl;
        cout << "HoughGrouping::run - Cell height: "  << lay->specificTopology().cellHeight()   << endl;
      }
      if ((*dtLayerIdIt).first.superLayer() == 2) continue;
      
      wirePosInLay     = LocalPoint(lay->specificTopology().wirePosition((*digiIt).wire()), 0, 0);
      wirePosGlob      = lay->toGlobal(wirePosInLay);
      wirePosInChamber = lay->chamber()->toLocal(wirePosGlob);
      
      if ((*dtLayerIdIt).first.superLayer() == 3) {
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()] = DTPrimitive();
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setTDCTime((*digiIt).time());
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap[(*dtLayerIdIt).first.layer() + 3][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      }
      else {
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()] = DTPrimitive();
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setTDCTime((*digiIt).time());
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setChannelId((*digiIt).wire());
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setLayerId((*dtLayerIdIt).first.layer());
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setSuperLayerId((*dtLayerIdIt).first.superLayer());
        digimap[(*dtLayerIdIt).first.layer() - 1][(*digiIt).wire()].setCameraId((*dtLayerIdIt).first.rawId());
      }
      
      // Obtaining geometrical info of the chosen chamber
      if (xlowlim == 0 && xhighlim == 0 && zlowlim == 0 && zhighlim == 0) {
        thewheel   = (*dtLayerIdIt).first.wheel();
        thestation = (*dtLayerIdIt).first.station();
        thesector  = (*dtLayerIdIt).first.sector();
        ObtainGeometricalBorders(lay);
      }
      
      if (debug) {
        cout << "HoughGrouping::run - X position of the cell (chamber frame of reference): " << wirePosInChamber.x() << endl;
        cout << "HoughGrouping::run - Y position of the cell (chamber frame of reference): " << wirePosInChamber.y() << endl;
        cout << "HoughGrouping::run - Z position of the cell (chamber frame of reference): " << wirePosInChamber.z() << endl;
      }
      
      hitvec.push_back( {wirePosInChamber.x() - 1.05, wirePosInChamber.z()} );
      hitvec.push_back( {wirePosInChamber.x() + 1.05, wirePosInChamber.z()} );
      nhits += 2;
      
      idigi++;
    }
  }
  
  if (debug) {
    cout << "\nHoughGrouping::run - nhits: " << nhits << endl;
    cout << "HoughGrouping::run - idigi: "   << idigi << endl;
  }
  
  if (hitvec.size() == 0) {
    if (debug) cout << "HoughGrouping::run - No digis present in this chamber." << endl;
    return;
  }
  
  // Perform the Hough transform of the inputs.
  DoHoughTransform();

  // Obtain the maxima
  maxima = GetMaximaVector();
  ResetPosElementsOfLinespace();

  if (maxima.size() == 0) {
    if (debug) cout << "HoughGrouping::run - No good maxima found in this event." << endl;
    return;
  }

  DTChamberId TheChambId(thewheel, thestation, thesector);
  const DTChamber* TheChamb = dtGeom->chamber(TheChambId);
  std::vector<std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*>> cands;

  for (UShort_t ican = 0; ican < maxima.size(); ican++) {
    if (debug) cout << "\nHoughGrouping::run - Candidate number: " << ican << endl;
    cands.push_back( AssociateHits(TheChamb, maxima.at(ican).first, maxima.at(ican).second) );
  }

  // Now we filter them:
  OrderAndFilter(cands, outMpath);
  if (debug) cout << "HoughGrouping::run - now we have our muonpaths! It has " << outMpath->size() << " elements" << endl;

  Short_t indi = cands.size() - 1;
  while (! cands.empty() ) {
    delete[] get<1>(cands.at(indi));
    delete[] get<2>(cands.at(indi));
    delete[] get<4>(cands.at(indi));
    delete[] get<5>(cands.at(indi));

    cands.pop_back();
    indi--;
  }
  std::vector<std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*>>().swap(cands);
  return;
}


void HoughGrouping::finish() {
  if (debug) cout << "HoughGrouping::finish" << endl;
  return;
}



// ============================================================================
// Other methods
// ============================================================================
void HoughGrouping::ResetAttributes() {
  if (debug) cout << "HoughGrouping::ResetAttributes" << endl;
  // std::vector's:
  maxima.clear();
//   cands.clear();
  hitvec.clear();
  
  // Integer-type variables:
  spacebins = 0;
  idigi     = 0;
  nhits     = 0;
  xlowlim   = 0;
  xhighlim  = 0;
  zlowlim   = 0;
  zhighlim  = 0;
  thestation= 0;
  thesector = 0;
  thewheel  = 0;
  
  // Arrays:
  // NOTE: linespace array is treated and reset separately
  
  // Maps (dictionaries):
  posmap.clear();
  for (UShort_t abslay = 0; abslay < 8; abslay++) digimap[abslay].clear();
}


void HoughGrouping::ResetPosElementsOfLinespace() {
  if (debug) cout << "HoughGrouping::ResetPosElementsOfLinespace" << endl;
  for (UShort_t ab = 0; ab < anglebins; ab++) {
    delete[] linespace[ab];
  }
}


void HoughGrouping::ObtainGeometricalBorders(const DTLayer* lay) {
  if (debug) cout << "HoughGrouping::ObtainGeometricalBorders" << endl;
  LocalPoint  FirstWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(lay->chamber()->superLayer(1)->layer(1)->specificTopology().firstChannel()), 0, 0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobal  = lay->chamber()->superLayer(1)->layer(1)->toGlobal(FirstWireLocal);
  LocalPoint  FirstWireLocalCh = lay->chamber()->toLocal(FirstWireGlobal);
  
  LocalPoint  LastWireLocal(lay->chamber()->superLayer(1)->layer(1)->specificTopology().wirePosition(lay->chamber()->superLayer(1)->layer(1)->specificTopology().lastChannel()), 0, 0);
  GlobalPoint LastWireGlobal  = lay->chamber()->superLayer(1)->layer(1)->toGlobal(LastWireLocal);
  LocalPoint  LastWireLocalCh = lay->chamber()->toLocal(LastWireGlobal);
  
//   UShort_t upsl = thestation == 4 ? 2 : 3;
  UShort_t upsl = thestation == 4 ? 3 : 3;
  if (debug) cout << "HoughGrouping::ObtainGeometricalBorders - uppersuperlayer: " << upsl << endl;
  
  LocalPoint  FirstWireLocalUp(lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().wirePosition(lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().firstChannel()), 0, 0);  // TAKING INFO FROM L1 OF SL1 OF THE CHOSEN CHAMBER
  GlobalPoint FirstWireGlobalUp  = lay->chamber()->superLayer(upsl)->layer(4)->toGlobal(FirstWireLocalUp);
  LocalPoint  FirstWireLocalChUp = lay->chamber()->toLocal(FirstWireGlobalUp);
  
  xlowlim   = FirstWireLocalCh.x()   - lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  xhighlim  = LastWireLocalCh.x()    + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellWidth() / 2;
  zlowlim   = FirstWireLocalChUp.z() - lay->chamber()->superLayer(upsl)->layer(4)->specificTopology().cellHeight() / 2;
  zhighlim  = LastWireLocalCh.z()    + lay->chamber()->superLayer(1)->layer(1)->specificTopology().cellHeight() / 2;
  
  spacebins = TMath::Nint(TMath::Abs(xhighlim - xlowlim) / posbinwidth);
}


void HoughGrouping::DoHoughTransform() {
  if (debug) cout << "HoughGrouping::DoHoughTransform" << endl;
  // First we want to obtain the number of bins in angle that we want. To do so, we will consider at first a maximum angle of
  // (in rad.) pi/2 - arctan(0.3) (i.e. ~73º) and a resolution (width of bin angle) of 2º.
  
  if (debug) {
    cout << "HoughGrouping::DoHoughTransform - maxrads: "       << maxrads        << endl;
    cout << "HoughGrouping::DoHoughTransform - minangle: "      << minangle       << endl;
    cout << "HoughGrouping::DoHoughTransform - halfanglebins: " << halfanglebins  << endl;
    cout << "HoughGrouping::DoHoughTransform - anglebins: "     << anglebins      << endl;
    cout << "HoughGrouping::DoHoughTransform - oneanglebin: "   << oneanglebin    << endl;
    cout << "HoughGrouping::DoHoughTransform - spacebins: "     << spacebins      << endl;
  }
  
  Double_t rho = 0, phi = 0, sbx = 0;
  // lowinitsb defines the center of the first bin in the distance dimension
//   Double_t lowinitsb = xlowlim;
  Double_t lowinitsb = xlowlim + posbinwidth/2;
  
  // Initialisation
  for (UShort_t ab = 0; ab < anglebins; ab++) {
    linespace[ab] = new UShort_t[spacebins];
    sbx           = lowinitsb;
    for (UShort_t sb = 0; sb < spacebins; sb++) {
      posmap[sb]        = sbx;
      linespace[ab][sb] = 0;
      sbx               += posbinwidth;
    }
  }
  
  // Filling of the double array and actually doing the transform
  for (UShort_t i = 0; i < hitvec.size(); i++) {
    for (UShort_t ab = 0; ab < anglebins; ab++) {
      phi = anglemap[ab];
      rho = hitvec.at(i).first * TMath::Cos(phi) + hitvec.at(i).second * TMath::Sin(phi);
      sbx = lowinitsb - posbinwidth/2;
      for (UShort_t sb = 0; sb < spacebins; sb++) {
        if ( rho < sbx ) {
          linespace[ab][sb]++;
          break;
        }
        sbx += posbinwidth;
      }
    }
  }
}


std::vector<std::pair<Double_t, Double_t>> HoughGrouping::GetMaximaVector() {
  if (debug) cout << "HoughGrouping::GetMaximaVector" << endl;
  std::vector<std::tuple<Double_t, Double_t, UShort_t>> tmpvec; tmpvec.clear();
  
  Bool_t flagsearched = false; UShort_t numbertolookat = UpperNumber;
  
  while (!flagsearched) {
    for (UShort_t ab = 0; ab < anglebins; ab++) {
      for (UShort_t sb = 0; sb < spacebins; sb++) {
        if (linespace[ab][sb] >= numbertolookat) tmpvec.push_back({anglemap[ab], posmap[sb], linespace[ab][sb]});
      }
    }
    if (((numbertolookat - 1) < LowerNumber) || (tmpvec.size() > 0)) flagsearched = true;
    else                                                             numbertolookat--;
  }
  
  if (tmpvec.size() == 0) {
    if (debug) cout << "HoughGrouping::GetMaximaVector - No maxima could be found" << endl;
    std::vector<std::pair<Double_t, Double_t>> finalvec; finalvec.clear();
    return finalvec;
  }
  else {
    std::vector<std::pair<Double_t, Double_t>> finalvec = FindTheMaxima(tmpvec);
    
  //   cout << "\nAFTER" << endl;
  //   for (UShort_t i = 0; i < finalvec.size(); i++) {
  //     if (finalvec.at(i).first < TMath::PiOver2()) cout << "   ang (rad): " << finalvec.at(i).first << ", ang (º): " << finalvec.at(i).first * 360/TMath::TwoPi()       << ", pos: " << finalvec.at(i).second << endl;
  //     else                                         cout << "   ang (rad): " << finalvec.at(i).first << ", ang (º): " << 180 - finalvec.at(i).first * 360/TMath::TwoPi() << ", pos: " << finalvec.at(i).second << endl;
  //   }
    // And now obtain the values of m and n of the lines.
    for (UShort_t i = 0; i < finalvec.size(); i++) finalvec.at(i) = TransformPair(finalvec.at(i));
    return finalvec;
  }
}


std::pair<Double_t, Double_t> HoughGrouping::TransformPair(std::pair<Double_t, Double_t> inputpair) {
  if (debug) cout << "HoughGrouping::TransformPair" << endl;
  // input: (ang, pos); output: (m, n)
  if (inputpair.first == 0) return {1000,                            -1000 * inputpair.second};
  else                      return {-1./TMath::Tan(inputpair.first), inputpair.second/TMath::Sin(inputpair.first)};
}


std::vector<std::pair<Double_t, Double_t>> HoughGrouping::FindTheMaxima(std::vector<std::tuple<Double_t, Double_t, UShort_t>> inputvec) {
  if (debug) cout << "HoughGrouping::FindTheMaxima" << endl;
  Bool_t   fullyreduced = false;
  UShort_t ind          = 0;
  
  std::vector<UShort_t> chosenvec; chosenvec.clear();
  std::vector<std::pair<Double_t, Double_t>> resultvec; resultvec.clear();
  std::pair<Double_t, Double_t> finalpair = {};
  
  if (debug) cout << "HoughGrouping::FindTheMaxima - prewhile" << endl;
  while (!fullyreduced) {
    if (debug) {
      cout << "\nHoughGrouping::FindTheMaxima - New iteration" << endl;
      cout << "HoughGrouping::FindTheMaxima - inputvec size: " << inputvec.size() << endl;
      cout << "HoughGrouping::FindTheMaxima - ind: " << ind << endl;
      cout << "HoughGrouping::FindTheMaxima - maximum deltaang: " << maxdeltaAng << " and maximum deltapos: " << maxdeltaPos << endl;
    }
    chosenvec.clear();
    //calculate distances and check out the ones that are near
    if (debug) cout << "HoughGrouping::FindTheMaxima - Ours have " << get<2>(inputvec.at(ind)) << " entries, ang.: " << get<0>(inputvec.at(ind)) << " and pos.: " << get<1>(inputvec.at(ind)) << endl;
    
    for (UShort_t j = ind + 1; j < inputvec.size(); j++) {
      if (GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).first <= maxdeltaAng && GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).second <= maxdeltaPos ) {
        chosenvec.push_back(j);
        if (debug) cout << "HoughGrouping::FindTheMaxima -     - Adding num.  " << j << " with deltaang: " <<  GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).first << ", and deltapos: " << GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).second << " and with " << get<2>(inputvec.at(j)) << " entries, ang.: " << get<0>(inputvec.at(j)) << " and pos.: " << get<1>(inputvec.at(j)) << endl;
      }
      else if (debug) cout << "HoughGrouping::FindTheMaxima -     - Ignoring num. " << j << " with deltaang: " <<  GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).first << ", and deltapos: " << GetTwoDelta( inputvec.at(ind), inputvec.at(j) ).second << " and with " << get<2>(inputvec.at(j)) << " entries." << endl;
    }
    
    if (debug) cout << "HoughGrouping::FindTheMaxima - chosenvecsize: " << chosenvec.size() << endl;
    
    if (chosenvec.size() == 0) {
      if (ind + 1 >= (UShort_t)inputvec.size()) fullyreduced = true;
      if ((get<0>(inputvec.at(ind)) <= maxrads) || (get<0>(inputvec.at(ind)) >= TMath::Pi() - maxrads)) resultvec.push_back({get<0>(inputvec.at(ind)), get<1>(inputvec.at(ind))});
      else if (debug)                                                                                    cout << "HoughGrouping::FindTheMaxima -     - Candidate dropped due to an excess in angle" << endl;
      ind++;
      continue;
    }
    
    // Now average them
    finalpair = GetAveragePoint(inputvec, ind, chosenvec);
    
    // Erase the ones you used
    inputvec.erase(inputvec.begin() + ind);
    for (Short_t j = chosenvec.size() - 1; j > -1; j--) {
      if (debug) cout << "HoughGrouping::FindTheMaxima - erasing index: " << chosenvec.at(j) - 1 << endl;
      inputvec.erase(inputvec.begin() + chosenvec.at(j) - 1);
    }
    
    if (debug) cout << "HoughGrouping::FindTheMaxima - inputvec size: " << inputvec.size() << endl;
    
    // And add the one you calculated:
    if ((finalpair.first <= maxrads) || (finalpair.first >= TMath::Pi() - maxrads)) resultvec.push_back(finalpair);
    else if (debug)                                                                 cout << "HoughGrouping::FindTheMaxima -     - Candidate dropped due to an excess in angle" << endl;
    
    if (ind + 1 >= (UShort_t)inputvec.size()) fullyreduced = true;
    if (debug) cout << "HoughGrouping::FindTheMaxima - iteration ends" << endl;
    ind++;
  }
  if (debug) cout << "HoughGrouping::FindTheMaxima - postwhile" << endl;
  return resultvec;
}


std::pair<Double_t, Double_t> HoughGrouping::GetTwoDelta(std::tuple<Double_t, Double_t, UShort_t> pair1, std::tuple<Double_t, Double_t, UShort_t> pair2) {
  if (debug) cout << "HoughGrouping::GetTwoDelta" << endl;
  return {TMath::Abs(get<0>(pair1) - get<0>(pair2)), TMath::Abs(get<1>(pair1) - get<1>(pair2))};
}


std::pair<Double_t, Double_t> HoughGrouping::GetAveragePoint(std::vector<std::tuple<Double_t, Double_t, UShort_t>> inputvec, UShort_t firstindex, std::vector<UShort_t> indexlist) {
  if (debug) cout << "HoughGrouping::GetAveragePoint" << endl;
  std::vector<Double_t> xs; xs.clear();
  std::vector<Double_t> ys; ys.clear();
  std::vector<Double_t> ws; ws.clear();
  xs.push_back(get<0>(inputvec.at(firstindex))); ys.push_back(get<1>(inputvec.at(firstindex)));
//   ws.push_back(get<2>(inputvec.at(firstindex)));
  ws.push_back(TMath::Exp(get<2>(inputvec.at(firstindex))));
  for (UShort_t i = 0; i < indexlist.size(); i++) {
    xs.push_back(get<0>(inputvec.at(indexlist.at(i)))); ys.push_back(get<1>(inputvec.at(indexlist.at(i))));
//     ws.push_back(get<2>(inputvec.at(indexlist.at(i))));
    ws.push_back(TMath::Exp(get<2>(inputvec.at(indexlist.at(i)))));
  }
  return {TMath::Mean(xs.begin(), xs.end(), ws.begin()), TMath::Mean(ys.begin(), ys.end(), ws.begin())};
}


std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> HoughGrouping::AssociateHits(const DTChamber* thechamb, Double_t m, Double_t n) {
  if (debug) cout << "HoughGrouping::AssociateHits" << endl;
  LocalPoint  tmpLocal, AWireLocal, AWireLocalCh, tmpLocalCh, thepoint;
  GlobalPoint tmpGlobal, AWireGlobal;
  Double_t tmpx = 0; Double_t distleft = 0; Double_t distright = 0;
  UShort_t tmpwire = 0; UShort_t abslay = 0; LATERAL_CASES lat = NONE;
  Bool_t isleft = false; Bool_t isright = false;
  
  std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> returntuple;
  get<0>(returntuple) = 0;
  get<1>(returntuple) = new Bool_t[8];
  get<2>(returntuple) = new Bool_t[8];
  get<3>(returntuple) = 0;
  get<4>(returntuple) = new Double_t[8];
  for (UShort_t lay = 0; lay < 8; lay++) {
    get<1>(returntuple)[lay] = false;
    get<2>(returntuple)[lay] = false;
    get<4>(returntuple)[lay] = 0.;
  }
  get<5>(returntuple) = new DTPrimitive[8];
  // 0: # of layers with hits.
  // 1: # of hits of high quality (the expected line crosses the cell).
  // 2: # of hits of low quality (the expected line is in a neighbouring cell).
  // 3: absolute diff. between the number of hits in SL1 and SL3.
  // 4: absolute distance to all hits of the segment.
  // 5: DTPrimitive of the candidate.
  
  if (debug) cout << "HoughGrouping::AssociateHits - Beginning SL loop" << endl;
  for (UShort_t sl = 1; sl < 3 + 1; sl++) {
    if (sl == 2) continue;
    if (debug) cout << "HoughGrouping::AssociateHits - SL: " << sl << endl;
    
    for (UShort_t l = 1; l < 4 + 1; l++) {
      if (debug) cout << "HoughGrouping::AssociateHits - L: " << l << endl;
      isleft = false; isright = false; lat = NONE; distleft = 0; distright = 0;
      if (sl == 1) abslay  = l - 1;
      else         abslay  = l + 3;
      AWireLocal   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel()), 0, 0);
      AWireGlobal  = thechamb->superLayer(sl)->layer(l)->toGlobal(AWireLocal);
      AWireLocalCh = thechamb->toLocal(AWireGlobal);
      tmpx = (AWireLocalCh.z() - n) / m;
      
      if ((tmpx <= xlowlim) || (tmpx >= xhighlim)) {
        get<5>(returntuple)[abslay] = DTPrimitive(); // empty primitive
        continue;
      }
      
      thepoint = LocalPoint(tmpx, 0, AWireLocalCh.z());
      tmpwire  = thechamb->superLayer(sl)->layer(l)->specificTopology().channel(thepoint);
      if (debug) cout << "HoughGrouping::AssociateHits - Wire number: "  << tmpwire << endl;
      if (debug) cout << "HoughGrouping::AssociateHits - First channel in layer: "  << thechamb->superLayer(sl)->layer(l)->specificTopology().firstChannel() << endl;
      if ((digimap[abslay]).count(tmpwire)) {
        // OK, we have a digi, let's choose the laterality, if we can:
        tmpLocal   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire), 0, 0);
        tmpGlobal  = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
        tmpLocalCh = thechamb->toLocal(tmpGlobal);
        
        if (TMath::Abs(tmpLocalCh.x() - thepoint.x()) >= MaxDistanceToWire) {
          // The distance where lateralities are not put is 0.03 cm, which is a conservative threshold for the resolution of the cells.
          if   ((tmpLocalCh.x() - thepoint.x()) > 0) lat = LEFT;
          else                                       lat = RIGHT;
        }
        
        // Filling info
        get<0>(returntuple)++;
        get<1>(returntuple)[abslay] = true;
        get<2>(returntuple)[abslay] = true;
        if      (lat == LEFT)  get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() - 1.05));
        else if (lat == RIGHT) get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() + 1.05));
        else                   get<4>(returntuple)[abslay] = TMath::Abs(tmpx - tmpLocalCh.x());
        get<5>(returntuple)[abslay] = DTPrimitive(digimap[abslay][tmpwire]);
        get<5>(returntuple)[abslay].setLaterality(lat);
      }
      else {
        if (debug) cout << "HoughGrouping::AssociateHits - No hit in the crossing cell" << endl;
        if ((digimap[abslay]).count(tmpwire - 1)) isleft  = true;
        if ((digimap[abslay]).count(tmpwire + 1)) isright = true;
        if (debug) cout << "HoughGrouping::AssociateHits - There is in the left: "  << (Int_t)isleft  << endl;
        if (debug) cout << "HoughGrouping::AssociateHits - There is in the right: " << (Int_t)isright << endl;
        
        if ((isleft) && (!isright)) {
          tmpLocal   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          tmpGlobal  = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);
          
          // Filling info
          get<0>(returntuple)++;
          get<2>(returntuple)[abslay] = true;
          get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() + 1.05));
          get<5>(returntuple)[abslay] = DTPrimitive(digimap[abslay][tmpwire - 1]);
          get<5>(returntuple)[abslay].setLaterality(RIGHT);
        }
        else if ((!isleft) && (isright)) {
          tmpLocal   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          tmpGlobal  = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal);
          tmpLocalCh = thechamb->toLocal(tmpGlobal);
          
          // Filling info
          get<0>(returntuple)++;
          get<2>(returntuple)[abslay] = true;
          get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() - 1.05));
          get<5>(returntuple)[abslay] = DTPrimitive(digimap[abslay][tmpwire + 1]);
          get<5>(returntuple)[abslay].setLaterality(LEFT);
        }
        else if ((isleft) && (isright)) {
          LocalPoint  tmpLocal_l   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire - 1), 0, 0);
          GlobalPoint tmpGlobal_l  = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_l);
          LocalPoint  tmpLocalCh_l = thechamb->toLocal(tmpGlobal_l);
          
          LocalPoint  tmpLocal_r   = LocalPoint(thechamb->superLayer(sl)->layer(l)->specificTopology().wirePosition(tmpwire + 1), 0, 0);
          GlobalPoint tmpGlobal_r  = thechamb->superLayer(sl)->layer(l)->toGlobal(tmpLocal_r);
          LocalPoint  tmpLocalCh_r = thechamb->toLocal(tmpGlobal_r);
          
          distleft = TMath::Abs(thepoint.x() - tmpLocalCh_l.x() ); distright = TMath::Abs(thepoint.x() - tmpLocalCh_r.x() );
          
          // Filling info
          get<0>(returntuple)++;
          get<2>(returntuple)[abslay] = true;
          if (distleft < distright) {
            get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() + 1.05));
            get<5>(returntuple)[abslay] = DTPrimitive(digimap[abslay][tmpwire - 1]);
            get<5>(returntuple)[abslay].setLaterality(RIGHT);
          }
          else {
            get<4>(returntuple)[abslay] = TMath::Abs(tmpx - (tmpLocalCh.x() - 1.05));
            get<5>(returntuple)[abslay] = DTPrimitive(digimap[abslay][tmpwire + 1]);
            get<5>(returntuple)[abslay].setLaterality(LEFT);
          }
        }
        else { // case where there are no digis
          get<5>(returntuple)[abslay] = DTPrimitive(); // empty primitive
        }
      }
    }
  }
  
  SetDifferenceBetweenSL(returntuple);
  if (debug) {
    cout << "HoughGrouping::AssociateHits - Finishing with the candidate. We have found the following of it:" << endl;
    cout << "HoughGrouping::AssociateHits - # of layers with hits: "               << get<0>(returntuple) << endl;
    for (UShort_t lay = 0; lay < 8; lay++) {
      cout << "HoughGrouping::AssociateHits - For absolute layer: "                  << lay << endl;
      cout << "HoughGrouping::AssociateHits - # of HQ hits: "                        << get<1>(returntuple)[lay] << endl;
      cout << "HoughGrouping::AssociateHits - # of LQ hits: "                        << get<2>(returntuple)[lay] << endl;
    }
    cout << "HoughGrouping::AssociateHits - Abs. diff. between SL1 and SL3 hits: " << get<3>(returntuple) << endl;
    for (UShort_t lay = 0; lay < 8; lay++) {
      cout << "HoughGrouping::AssociateHits - For absolute layer: "                  << lay << endl;
      cout << "HoughGrouping::AssociateHits - Abs. distance to digi: "               << get<4>(returntuple)[lay] << endl;
    }
  }
  return returntuple;
}


void HoughGrouping::SetDifferenceBetweenSL(std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> &tupl) {
  if (debug) cout << "HoughGrouping::SetDifferenceBetweenSL" << endl;
  Short_t absres = 0;
  for (UShort_t lay = 0; lay < 8; lay++) {
    if ((get<5>(tupl))[lay].getChannelId() > 0) {
      if (lay <= 3) absres++;
      else          absres--;
    }
  }
  
  if (absres >= 0) get<3>(tupl) = absres;
  else             get<3>(tupl) = (UShort_t) (-absres);
}


void HoughGrouping::OrderAndFilter(std::vector<std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*>> &invector, std::vector<MuonPath*> *&outMuonPath) {
  if (debug) cout << "HoughGrouping::OrderAndFilter" << endl;
  // 0: # of layers with hits.
  // 1: # of hits of high quality (the expected line crosses the cell).
  // 2: # of hits of low quality (the expected line is in a neighbouring cell).
  // 3: absolute diff. between the number of hits in SL1 and SL3.
  // 4: absolute distance to all hits of the segment.
  // 5: DTPrimitive of the candidate.
  
  std::vector<UShort_t>  elstoremove; elstoremove.clear();
  // Ordering:
  if (debug) cout << "HoughGrouping::OrderAndFilter - First ordering" << endl;
  std::sort(invector.begin(), invector.end(), HoughOrdering);
  
  // Now filtering:
  UShort_t ind = 0; Bool_t filtered = false;
  if (debug) cout << "HoughGrouping::OrderAndFilter - Entering while" << endl;
  while (!filtered) {
    if (debug) cout << "\nHoughGrouping::OrderAndFilter - New iteration with ind: " << ind << endl;
    elstoremove.clear();
    for (UShort_t i = ind + 1; i < invector.size(); i++) {
      if (debug) cout << "HoughGrouping::OrderAndFilter - Checking index: " << i << endl;
      for (UShort_t lay = 0; lay < 8; lay++) {
        if (debug) cout << "HoughGrouping::OrderAndFilter - Checking layer number: " << lay << endl;
        if ((get<5>(invector.at(i))[lay].getChannelId() == get<5>(invector.at(ind))[lay].getChannelId()) && (get<5>(invector.at(ind))[lay].getChannelId() != -1)) {
          get<0>(invector.at(i))--;
          get<1>(invector.at(i))[lay] = false;
          get<2>(invector.at(i))[lay] = false;
          SetDifferenceBetweenSL(invector.at(i));
          // We check that if its a different laterality, the best candidate of the two of them changes its laterality to not-known (that is, both).
          if (get<5>(invector.at(i))[lay].getLaterality() != get<5>(invector.at(ind))[lay].getLaterality()) get<5>(invector.at(ind))[lay].setLaterality(NONE);
          get<5>(invector.at(i))[lay] = DTPrimitive();
        }
      }
      if (debug) cout << "HoughGrouping::OrderAndFilter - Finished checking all the layers, now seeing if we should remove the candidate" << endl;
      
      if (! AreThereEnoughHits(invector.at(i)) ) {
        if (debug) cout << "HoughGrouping::OrderAndFilter - This candidate shall be removed!" << endl;
        elstoremove.push_back((UShort_t)i);
      }
    }
    
    if (debug) cout << "HoughGrouping::OrderAndFilter - We are gonna erase " << elstoremove.size() << " elements" << endl;
    
    for (Short_t el = (elstoremove.size() - 1); el > -1; el--) {
      delete[] get<1>(invector.at(elstoremove.at(el)));
      delete[] get<2>(invector.at(elstoremove.at(el)));
      delete[] get<4>(invector.at(elstoremove.at(el)));
      delete[] get<5>(invector.at(elstoremove.at(el)));
      invector.erase(invector.begin() + elstoremove.at(el));
    }
    
    if (ind + 1 == (UShort_t)invector.size()) filtered = true;
    else                                      std::sort(invector.begin() + ind + 1, invector.end(), HoughOrdering);
    ind++;
  }
  
  // Ultimate filter: if the remaining do not fill the requirements (configurable through pset arguments), they are removed also.
  for (Short_t el = (invector.size() - 1); el > -1; el--) {
    if (! AreThereEnoughHits(invector.at(el))) {
      delete[] get<1>(invector.at(el));
      delete[] get<2>(invector.at(el));
      delete[] get<4>(invector.at(el));
      delete[] get<5>(invector.at(el));
      invector.erase(invector.begin() + el);
    }
  }
  
  if (invector.size() == 0) {
    if (debug) cout << "HoughGrouping::OrderAndFilter - We do not have candidates with the minimum hits required." << endl;
    return;
  }
  else if (debug) cout << "HoughGrouping::OrderAndFilter - At the end, we have only " << invector.size() << " good paths!" << endl;
  
  // Packing dt primitives
  for (UShort_t i = 0; i < invector.size(); i++) {
    DTPrimitive *ptrPrimitive[8];
    UShort_t tmplowfill = 0; UShort_t tmpupfill = 0;
    for (UShort_t lay = 0; lay < 8; lay++) {
      ptrPrimitive[lay] = new DTPrimitive(get<5>(invector.at(i))[lay]);
      if (debug) {
        cout << "\nHoughGrouping::OrderAndFilter - cameraid: "       << ptrPrimitive[lay]->getCameraId() << endl;
        cout << "HoughGrouping::OrderAndFilter - channelid (GOOD): " << ptrPrimitive[lay]->getChannelId() << endl;
        cout << "HoughGrouping::OrderAndFilter - channelid (AM):   " << ptrPrimitive[lay]->getChannelId() - 1 << endl;
      }
      // Fixing channel ID to AM conventions...
      if (ptrPrimitive[lay]->getChannelId() != -1) ptrPrimitive[lay]->setChannelId(ptrPrimitive[lay]->getChannelId() - 1);

      if (ptrPrimitive[lay]->getCameraId() > 0) {
        if (lay < 4) tmplowfill++;
        else         tmpupfill++;
      }
    }
    
    MuonPath *ptrMuonPath = new MuonPath(ptrPrimitive, tmplowfill, tmpupfill);
//     MuonPath *ptrMuonPath = new MuonPath(ptrPrimitive, 7);
    outMuonPath->push_back(ptrMuonPath);
    if (debug) {
      for (UShort_t lay = 0; lay < 8; lay++) {
        cout << "HoughGrouping::OrderAndFilter - Final cameraID: "  << outMuonPath->back()->getPrimitive(lay)->getCameraId()  << endl;
        cout << "HoughGrouping::OrderAndFilter - Final channelID: " << outMuonPath->back()->getPrimitive(lay)->getChannelId() << endl;
        cout << "HoughGrouping::OrderAndFilter - Final time: "      << outMuonPath->back()->getPrimitive(lay)->getTDCTime()   << endl;
      }
    }
  }
  return;
}


Bool_t HoughGrouping::AreThereEnoughHits(std::tuple<UShort_t, Bool_t*, Bool_t*, UShort_t, Double_t*, DTPrimitive*> tupl) {
  if (debug) cout << "HoughGrouping::AreThereEnoughHits" << endl;
  UShort_t numhitssl1 = 0; UShort_t numhitssl3 = 0;
  for (UShort_t lay = 0; lay < 8; lay++) {
    if ((get<5>(tupl)[lay].getChannelId() > 0) && (lay < 4)) numhitssl1++;
    else if (get<5>(tupl)[lay].getChannelId() > 0)           numhitssl3++;
  }
  
  if (debug) cout << "HoughGrouping::AreThereEnoughHits - Hits in SL1: " << numhitssl1 << endl;
  if (debug) cout << "HoughGrouping::AreThereEnoughHits - Hits in SL3: " << numhitssl3 << endl;
  
  if ((numhitssl1 != 0) && (numhitssl3 != 0)) { // Correlated candidates
    if ((numhitssl1 + numhitssl3) >= minNLayerHits) {
      if      (numhitssl1 > numhitssl3) {
        return ((numhitssl1 >= minSingleSLHitsMax) && (numhitssl3 >= minSingleSLHitsMin));
      }
      else if (numhitssl3 > numhitssl1) {
        return ((numhitssl3 >= minSingleSLHitsMax) && (numhitssl1 >= minSingleSLHitsMin));
      }
      else return true;
    }
  }
  else if (allowUncorrelatedPatterns) {         // Uncorrelated candidates
    return ((numhitssl1 + numhitssl3) >= minNLayerHits);
  }
  else {
    return false;
  }
  return false;
}
