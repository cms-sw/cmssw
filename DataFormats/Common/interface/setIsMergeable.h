#ifndef DataFormats_Common_setIsMergeable_h
#define DataFormats_Common_setIsMergeable_h
// -*- C++ -*-
//
// Package:     Common
// Class  :     setIsMergeable
//
/*
 Description: Should be called only after the BranchDescription::init
              function has been called either directly or through
              the BranchDescription constructor.

              Helper function used to set the isMergeable data member
              in BranchDescription. It would be much more convenient
              to have been able to put this inside the BranchDescription
              class itself in the init function, but the WrapperBase
              class in package DataFormats/Common is needed to implement
              this and that cannot be used in package DataFormats/Provenance
              because it would create a circular dependency.

              Anything creating a BranchDescription or reading one from
              a ROOT file will need to call this directly if they need
              the isMergeable data member to be set properly. Note that
              the isMergeable data member will default to false so if
              you know there are no mergeable run or lumi products, calling
              this is unnecessary.
*/
//
// Original Author:  W. David Dagenhart
//         Created:  21 June 2018
//

namespace edm {

  class BranchDescription;

  void setIsMergeable(BranchDescription&);
}

#endif
