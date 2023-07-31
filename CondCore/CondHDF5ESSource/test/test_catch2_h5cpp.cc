// -*- C++ -*-
//
// Package:     CondCore/CondHDF5ESSource
// Class  :     test_catch2_h5cpp
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Christopher Jones
//         Created:  Fri, 30 Jun 2023 18:35:42 GMT
//

// system include files
#include <cassert>
#include "catch.hpp"

// user include files
#include "FWCore/Utilities/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CondCore/CondHDF5ESSource/plugins/h5_File.cc"
#include "CondCore/CondHDF5ESSource/plugins/h5_Group.cc"
#include "CondCore/CondHDF5ESSource/plugins/h5_DataSet.cc"
#include "CondCore/CondHDF5ESSource/plugins/h5_Attribute.cc"

namespace {
  std::string findFile(std::string const& iFile) { return edm::FileInPath::findFile(iFile); }
}  // namespace

TEST_CASE("test cms::h5 code", "[cms::h5]") {
  SECTION("File") {
    SECTION("good file") { REQUIRE_NOTHROW(cms::h5::File(findFile("test.h5"), cms::h5::File::kReadOnly)); }
    SECTION("missing file") { REQUIRE_THROWS_AS(cms::h5::File("missing", cms::h5::File::kReadOnly), cms::Exception); }
  }
  SECTION("Group") {
    cms::h5::File h5file(findFile("test.h5"), cms::h5::File::kReadOnly);

    SECTION("present") {
      std::shared_ptr<cms::h5::Group> g;
      REQUIRE_NOTHROW(g = h5file.findGroup("Agroup"));
      SECTION("name") { REQUIRE(g->name() == "/Agroup"); }
    }
    SECTION("missing") { REQUIRE_THROWS_AS(h5file.findGroup("missing"), cms::Exception); }
    SECTION("sub group") {
      auto a = h5file.findGroup("Agroup");
      REQUIRE_NOTHROW(a->findGroup("Bgroup"));
      REQUIRE(a->getNumObjs() == 5);  //group and 4 datasets
      REQUIRE(a->getObjnameByIdx(0) == "Bgroup");
    }
  }

  SECTION("DataSet") {
    cms::h5::File h5file(findFile("test.h5"), cms::h5::File::kReadOnly);
    {
      auto g = h5file.findGroup("Agroup");
      SECTION("missing") { REQUIRE_THROWS_AS(g->findDataSet("missing"), cms::Exception); }
      SECTION("byte array") {
        std::shared_ptr<cms::h5::DataSet> ds;
        REQUIRE_NOTHROW(ds = g->findDataSet("byte_array"));
        std::vector<char> b;
        REQUIRE_NOTHROW(b = ds->readBytes());
        REQUIRE(b.size() == 1);
        REQUIRE(b[0] == 1);
      }
    }
    SECTION("IOVSyncValue") {
      auto g = h5file.findGroup("SyncGroup");
      std::shared_ptr<cms::h5::DataSet> ds;
      REQUIRE_NOTHROW(ds = g->findDataSet("sync"));
      std::vector<cond::hdf5::IOVSyncValue> sv;
      REQUIRE_NOTHROW(sv = ds->readSyncValues());
      REQUIRE(sv.size() == 3);
      REQUIRE(sv[0].high_ == 1);
      REQUIRE(sv[0].low_ == 0);
      REQUIRE(sv[1].high_ == 2);
      REQUIRE(sv[1].low_ == 1);
      REQUIRE(sv[2].high_ == 0xFFFFFFFF);
      REQUIRE(sv[2].low_ == 0xFFFFFFFF);
    }
  }
  SECTION("refs") {
    cms::h5::File h5file(findFile("test.h5"), cms::h5::File::kReadOnly);

    std::shared_ptr<cms::h5::DataSet> ds;
    auto g = h5file.findGroup("RefGroup");
    SECTION("group") {
      REQUIRE_NOTHROW(ds = g->findDataSet("groupRefs"));
      std::vector<hobj_ref_t> r;
      REQUIRE_NOTHROW(r = ds->readRefs());
      REQUIRE(r.size() == 2);

      {
        auto deref_g = h5file.derefGroup(r[0]);
        REQUIRE(deref_g->name() == "/Agroup");
      }
      {
        auto deref_g = h5file.derefGroup(r[1]);
        REQUIRE(deref_g->name() == "/Agroup/Bgroup");
      }
    }
    SECTION("DataSet") {
      SECTION("refs") {
        REQUIRE_NOTHROW(ds = g->findDataSet("dsetRefs"));
        std::vector<hobj_ref_t> r;
        REQUIRE_NOTHROW(r = ds->readRefs());
        REQUIRE(r.size() == 1);
        auto ds = h5file.derefDataSet(r[0]);
        std::vector<char> b;
        REQUIRE_NOTHROW(b = ds->readBytes());
        REQUIRE(b.size() == 1);
        REQUIRE(b[0] == 1);
      }
      SECTION("refs 2D") {
        REQUIRE_NOTHROW(ds = g->findDataSet("dset2DRefs"));
        std::vector<hobj_ref_t> r;
        REQUIRE_NOTHROW(r = ds->readRefs());
        REQUIRE(r.size() == 4);
        for (size_t i = 0; i < r.size(); ++i) {
          auto ds = h5file.derefDataSet(r[i]);
          std::vector<char> b;
          REQUIRE_NOTHROW(b = ds->readBytes());
          REQUIRE(b.size() == i + 1);
          for (auto v : b) {
            REQUIRE(v == static_cast<char>(i + 1));
          }
        }
      }
    }
  }

  SECTION("Attribute") {
    cms::h5::File h5file(findFile("test.h5"), cms::h5::File::kReadOnly);
    SECTION("missing") { REQUIRE_THROWS_AS(h5file.findAttribute("missing"), cms::Exception); }
    SECTION("in file") {
      auto at = h5file.findAttribute("at");
      REQUIRE(at.get() != nullptr);
      REQUIRE(at->readString() == "fileAt");
    }
    SECTION("in group") {
      auto g = h5file.findGroup("Agroup");
      auto a = g->findAttribute("b_at");
      REQUIRE(a->readString() == "groupAt");
    }
  }
}
