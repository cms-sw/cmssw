// -*- C++ -*-
//
// Package:     Services
// Class  :     FixMissingStreamerInfos
//
// Implementation:

/** \class edm::service::FixMissingStreamerInfos

This service is used to open and close a ROOT file that contains
StreamerInfo objects causing them to be saved in memory. It is
used when reading a file written with a version of ROOT with a
bug that caused it to fail to write out StreamerInfo objects.
(see Issue 41246).

CMSSW_13_0_0 had such a problem and files were written with
this problem. When using this service to read files written
with this release set the "fileInPath" parameter to the string
"IOPool/Input/data/fileContainingStreamerInfos_13_0_0.root".
This file is saved in the cms-data repository for IOPool/Input.
Note that it was difficult to identify all the problem classes
and we might have missed some. If there are additional problem
classes a new version of this file can be generated with script
IOPool/Input/scripts/makeFileContainingStreamerInfos.C. If the
problem ever recurs in ROOT with a different release, one could
use that script to generate a file containing StreamerInfos for
other releases.

    \author W. David Dagenhart, created 30 October, 2023

*/

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/FileInPath.h"

#include "TFile.h"

namespace edm {
  namespace service {

    class FixMissingStreamerInfos {
    public:
      FixMissingStreamerInfos(ParameterSet const&, ActivityRegistry&);
      static void fillDescriptions(ConfigurationDescriptions&);

    private:
      FileInPath fileInPath_;
    };

    FixMissingStreamerInfos::FixMissingStreamerInfos(ParameterSet const& pset, edm::ActivityRegistry&)
        : fileInPath_(pset.getUntrackedParameter<FileInPath>("fileInPath")) {
      auto tFile = TFile::Open(fileInPath_.fullPath().c_str());
      if (!tFile || tFile->IsZombie()) {
        throw cms::Exception("FixMissingStreamerInfo")
            << "Failed opening file containing missing StreamerInfos: " << fileInPath_.fullPath();
      }
      tFile->Close();
    }

    void FixMissingStreamerInfos::fillDescriptions(ConfigurationDescriptions& descriptions) {
      ParameterSetDescription desc;
      desc.addUntracked<FileInPath>("fileInPath");
      descriptions.add("FixMissingStreamerInfos", desc);
    }
  }  // namespace service
}  // namespace edm

using namespace edm::service;
DEFINE_FWK_SERVICE(FixMissingStreamerInfos);
