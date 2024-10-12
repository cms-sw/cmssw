#include "L1Trigger/Phase2L1GMT/interface/TrackConverter.h"
#include <fstream>

using namespace Phase2L1GMT;

TrackConverter::TrackConverter(const edm::ParameterSet& iConfig) : verbose_(iConfig.getParameter<int>("verbose")) {}

std::vector<ConvertedTTTrack> TrackConverter::convertTracks(
    const std::vector<edm::Ptr<l1t::TrackerMuon::L1TTTrackType> >& tracks) {
  std::vector<ConvertedTTTrack> out;
  out.reserve(tracks.size());
  for (const auto& t : tracks)
    out.push_back(convert(t));
  return out;
}

ConvertedTTTrack TrackConverter::convert(const edm::Ptr<TTTrack<Ref_Phase2TrackerDigi_> >& track) {
  uint charge = (track->rInv() < 0) ? 1 : 0;
  ap_int<BITSTTCURV> curvature = ap_int<BITSTTCURV>(track->getRinvBits());
  ap_int<BITSPHI> phisec = ap_int<BITSPHI>(ap_int<BITSTTPHI>(track->getPhiBits()) / 2);
  ap_int<BITSTTTANL> tanLambda = ap_int<BITSTTTANL>(track->getTanlBits());
  ap_int<BITSZ0> z0 = ap_int<BITSZ0>(ap_int<BITSTTZ0>(track->getZ0Bits()) / (1 << (BITSTTZ0 - BITSZ0)));
  ap_int<BITSD0> d0 = ap_int<BITSD0>(ap_int<BITSTTD0>(track->getD0Bits()) / (1 << (BITSTTD0 - BITSD0)));
  //calculate pt
  ap_uint<BITSTTCURV - 1> absCurv =
      curvature > 0 ? ap_uint<BITSTTCURV - 1>(curvature) : ap_uint<BITSTTCURV - 1>(-curvature);
  ap_uint<BITSPT> pt = ptLUT[ptLookup(absCurv)];
  ap_uint<1> quality = generateQuality(track);
  ap_uint<BITSTTTANL - 1> absTanL =
      tanLambda > 0 ? ap_uint<BITSTTTANL - 1>(tanLambda) : ap_uint<BITSTTTANL - 1>(-tanLambda);
  ap_uint<BITSETA - 1> absEta = etaLUT[etaLookup(absTanL)];
  ap_int<BITSETA> eta = tanLambda > 0 ? ap_int<BITSETA>(absEta) : ap_int<BITSETA>(-absEta);

  ap_int<BITSPHI> phi = ap_int<BITSPHI>(phisec + track->phiSector() * 910);

  wordtype word = 0;
  int bstart = 0;
  bstart = wordconcat<wordtype>(word, bstart, curvature, BITSTTCURV);
  bstart = wordconcat<wordtype>(word, bstart, phi, BITSPHI);  //was phiSec
  bstart = wordconcat<wordtype>(word, bstart, tanLambda, BITSTTTANL);
  bstart = wordconcat<wordtype>(word, bstart, z0, BITSZ0);
  bstart = wordconcat<wordtype>(word, bstart, d0, BITSD0);
  bstart = wordconcat<wordtype>(word, bstart, uint(track->chi2()), 4);

  ConvertedTTTrack convertedTrack(charge, curvature, absEta, pt, eta, phi, z0, d0, quality, word);
  convertedTrack.setOfflineQuantities(LSBpt * pt, LSBeta * eta, LSBphi * phi);
  if (verbose_)
    convertedTrack.print();
  convertedTrack.setTrkPtr(track);

  //printouts
 /* 
  fstream outfile("/uscms/home/hancelin/testing/CMSSW_14_1_0_pre3/src/trackconverter_printouts.txt", ios::app);
  outfile << "Input track: ";
  
  outfile << "k=" + to_string(ap_int<BITSTTCURV>(track->getRinvBits())) + ", ";
  outfile << "phi=" + to_string(ap_int<BITSTTPHI>(track->getPhiBits())) + ", ";
  outfile << "tanL=" + to_string(ap_int<BITSTTTANL>(track->getTanlBits())) + ", ";
  outfile << "z0=" + to_string(ap_int<BITSTTZ0>(track->getZ0Bits())) + ", ";
  outfile << "d0=" + to_string(ap_int<BITSTTD0>(track->getD0Bits())) + ", ";
  outfile << "chi2RPhi=" + to_string(track->getChi2RPhiBits()) + ", ";
  outfile << "chi2RZ=" + to_string(track->getChi2RZBits()) + ", ";
  outfile << "bend_chi2=" + to_string(track->getBendChi2Bits()) + ", ";
  outfile << "hitmask=" + to_string(track->getHitPatternBits()) + ", ";
  outfile << "trackMVA=" + to_string(track->getMVAQualityBits()) + ", ";
  outfile << "otherMVA=" + to_string(track->getMVAOtherBits()) + ", ";
  outfile << "valid=" + to_string(track->getValidBits()) + ", ";
  outfile << "sector=" + to_string(track->phiSector()) + ", ";
  
  //outfile << "chi2RPhiBits=" + to_string(track->getChi2RPhiBits()) + ", ";
  //outfile << "chi2RZBits=" + to_string(track->getChi2RZBits()) + ", ";
  //outfile << "chi2RPhi=" + to_string(track->getChi2RPhi()) + ", ";
  //outfile << "chi2RZ=" + to_string(track->getChi2RZ()) + ", ";
  //outfile << "chi2=" + to_string(track->chi2()) + ", ";
  //outfile << "chi2z=" + to_string(track->chi2Z()) + ", ";
  //outfile << "chi2xy=" + to_string(track->chi2XY()) + ", ";
  //outfile << "chi2red=" + to_string(track->chi2Red()) + ", ";
  //outfile << "chi2zred=" + to_string(track->chi2ZRed()) + ", ";
  //outfile << "chi2xyred=" + to_string(track->chi2XYRed()) + ", ";
  //outfile << "sector=" + to_string(track->phiSector()) + ", ";
  //if (track->phiSector() > 8) { outfile << "AAAAAAAAAAAAAAAAAAA";}
  //outfile << "reserved=" + to_string(track->reserved()) + ", "; what is reserved????

  outfile << "\n";

  outfile << "Output track: ";
  //q, pt, phi, eta, z0, d0, quality
  outfile << "q=" + to_string(ap_uint<1>(convertedTrack.charge())) + ", ";
  outfile << "pt=" + to_string(ap_uint<BITSPT>(convertedTrack.pt())) + ", ";
  outfile << "phi=" + to_string(ap_int<BITSPHI>(convertedTrack.phi())) + ", ";
  outfile << "eta=" + to_string(ap_int<BITSETA>(convertedTrack.eta())) + ", ";
  outfile << "z0=" + to_string(ap_int<BITSZ0>(convertedTrack.z0())) + ", ";
  outfile << "d0=" + to_string(ap_int<BITSD0>(convertedTrack.d0())) + ", ";
  outfile << "quality=" + to_string(ap_uint<1>(convertedTrack.quality())) + ", ";

  outfile << "\n\n";
  outfile.close();
  */
  return convertedTrack;
}
