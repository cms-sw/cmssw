import argparse
import ROOT
from DataFormats.FWLite import Events, Handle
from L1Trigger.Phase2L1GT.l1tGTScales import scale_parameter


def object_name(object_type):
    if not hasattr(ROOT, "getObjectName"):
        ROOT.gInterpreter.Declare("""
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
const char* getObjectName(l1t::P2GTCandidate::ObjectType objectType){
    switch(objectType) {
        case l1t::P2GTCandidate::GCTNonIsoEg: return "GCTNonIsoEg";
        case l1t::P2GTCandidate::GCTIsoEg: return "GCTIsoEg";
        case l1t::P2GTCandidate::GCTJets: return "GCTJet";
        case l1t::P2GTCandidate::GCTTaus: return "GCTTau";
        case l1t::P2GTCandidate::GCTHtSum: return "GCTHtSum";
        case l1t::P2GTCandidate::GCTEtSum: return "GCTEtSum";
        case l1t::P2GTCandidate::GMTSaPromptMuons: return "GMTSaPromptMuon";
        case l1t::P2GTCandidate::GMTSaDisplacedMuons: return "GMTSaDisplacedMuon";
        case l1t::P2GTCandidate::GMTTkMuons: return "GMTTkMuon";
        case l1t::P2GTCandidate::GMTTopo: return "GMTTopo";
        case l1t::P2GTCandidate::GTTPromptJets: return "GTTPromptJet";
        case l1t::P2GTCandidate::GTTDisplacedJets: return "GTTDisplacedJet";
        case l1t::P2GTCandidate::GTTPhiCandidates: return "GTTPhiCandidate";
        case l1t::P2GTCandidate::GTTRhoCandidates: return "GTTRhoCandidate";
        case l1t::P2GTCandidate::GTTBsCandidates: return "GTTBsCandidate";
        case l1t::P2GTCandidate::GTTHadronicTaus: return "GTTHadronicTau";
        case l1t::P2GTCandidate::GTTPrimaryVert: return "GTTPrimaryVert";
        case l1t::P2GTCandidate::GTTPromptHtSum: return "GTTPromptHtSum";
        case l1t::P2GTCandidate::GTTDisplacedHtSum: return "GTTDisplacedHtSum";
        case l1t::P2GTCandidate::GTTEtSum: return "GTTEtSum";
        case l1t::P2GTCandidate::CL2Jets: return "CL2Jet";
        case l1t::P2GTCandidate::CL2Taus: return "CL2Tau";
        case l1t::P2GTCandidate::CL2Electrons: return "CL2Electron";
        case l1t::P2GTCandidate::CL2Photons: return "CL2Photon";
        case l1t::P2GTCandidate::CL2HtSum: return "CL2HtSum";
        case l1t::P2GTCandidate::CL2EtSum: return "CL2EtSum"; 
        default: return "Undefined";
    }
}
""")
    return ROOT.getObjectName(object_type)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='L1GT analyzer')
    parser.add_argument('in_filename', nargs="+", help='input filename')
    parser.add_argument('--prefix', '-p', default='file:', help='file prefix')
    parser.add_argument('--process', '-P', default='', help='Process to analyze')

    args = parser.parse_args()

    in_filenames_with_prefix = ['{}{}'.format(args.prefix, x) for x in args.in_filename]
    events = Events(in_filenames_with_prefix)

    print("number of events", events.size())
    print('*' * 80)

    for idx, event in enumerate(events):
        print('Event:', idx)

        algo_blocks = Handle('l1t::P2GTAlgoBlockCollection')
        event.getByLabel('l1tGTAlgoBlockProducer', '', args.process, algo_blocks)

        for algo_blk in algo_blocks.product():
            print(algo_blk.algoName(), algo_blk.decisionBeforeBxMaskAndPrescale())

            for obj in algo_blk.trigObjects():
                if object_name(obj.objectType()) in ["CL2Electron", "CL2Photon"]:
                    print(" {}: pt {:3.1f} eta {:3.2f} phi {:3.2f} iso: {:3.2f} relIso: {:3.2f}".format(
                        object_name(obj.objectType()), obj.pt(), obj.eta(), obj.phi(),
                        obj.hwIso() * scale_parameter.isolation_lsb.value(),
                        obj.hwIso() * scale_parameter.isolation_lsb.value()/(obj.hwPT() * scale_parameter.pT_lsb.value())))
                elif "Sum" not in object_name(obj.objectType()):
                    print(" {}: pt {:3.1f} eta {:3.2f} phi {:3.2f}".format(
                        object_name(obj.objectType()), obj.pt(), obj.eta(), obj.phi()))
                else:
                    print(" {}: pt {:3.1f} phi {:3.2f}".format(
                        object_name(obj.objectType()), obj.pt(), obj.phi()))

        print('*' * 80)
