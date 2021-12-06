// -*- C++ -*-
//
// Package:     Histograms
// Class  :     metoemdformat_t
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Tue Jul 21 11:00:06 CDT 2009
//

// system include files

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

// user include files
#include "DataFormats/Histograms/interface/MonitorElementCollection.h"

TEST_CASE("MonitorElementData::Path", "[MonitorElementData_Path]") {
  MonitorElementData::Path p;

  SECTION("Canonical Paths") {
    SECTION("short path") {
      std::string pathName = "Foo/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == pathName);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == pathName);
    }

    SECTION("2 dirs") {
      std::string pathName = "Foo/Bar/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == pathName);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == pathName);
    }

    SECTION("long path") {
      std::string pathName = "This/Is/A/Very/Long/Path/Name/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == pathName);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == pathName);
    }
  }
  SECTION("extra /") {
    SECTION("only /") {
      std::string pathName = "/";
      std::string canonical = "";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("only //") {
      std::string pathName = "//";
      std::string canonical = "";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("leading /") {
      std::string pathName = "/Foo/";
      std::string canonical = "Foo/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("ending //") {
      std::string pathName = "Foo//";
      std::string canonical = "Foo/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("middle //") {
      std::string pathName = "Foo//Bar/";
      std::string canonical = "Foo/Bar/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
  }
  SECTION("missing end /") {
    SECTION("blank") {
      std::string pathName = "";
      std::string canonical = "";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("one dir") {
      std::string pathName = "Foo";
      std::string canonical = "Foo/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
    SECTION("two dir") {
      std::string pathName = "Foo/Bar";
      std::string canonical = "Foo/Bar/";
      p.set(pathName, MonitorElementData::Path::Type::DIR);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname().empty());
      REQUIRE(p.getFullname() == canonical);
    }
  }
  SECTION("up dir") {
    SECTION("to beginning") {
      SECTION("end /") {
        std::string pathName = "Foo/../";
        std::string canonical = "";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
      SECTION("missing end /") {
        std::string pathName = "Foo/..";
        std::string canonical = "";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
    }
    SECTION("beyond beginning") {
      SECTION("end /") {
        std::string pathName = "Foo/../..";
        std::string canonical = "";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
      SECTION("missing end /") {
        std::string pathName = "Foo/../..";
        std::string canonical = "";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
    }
    SECTION("middle") {
      SECTION("back to beginning") {
        std::string pathName = "Foo/../Bar/";
        std::string canonical = "Bar/";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
      SECTION("midway") {
        std::string pathName = "Foo/Bar/../Biz/";
        std::string canonical = "Foo/Biz/";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
      SECTION("last") {
        std::string pathName = "Foo/Bar/../";
        std::string canonical = "Foo/";
        p.set(pathName, MonitorElementData::Path::Type::DIR);
        REQUIRE(p.getDirname() == canonical);
        REQUIRE(p.getObjectname().empty());
        REQUIRE(p.getFullname() == canonical);
      }
    }
    SECTION("multiple consecutive") {
      SECTION("two") {
        SECTION("back to beginning") {
          std::string pathName = "Foo/Biz/../../Bar/";
          std::string canonical = "Bar/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
        SECTION("midway") {
          std::string pathName = "Foo/Bar/Blah/../../Biz/";
          std::string canonical = "Foo/Biz/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
        SECTION("last") {
          std::string pathName = "Foo/Bar/Blah/../../";
          std::string canonical = "Foo/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
      }
      SECTION("three") {
        SECTION("back to beginning") {
          std::string pathName = "Foo/Biz/Bleep/../../../Bar/";
          std::string canonical = "Bar/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
        SECTION("midway") {
          std::string pathName = "Foo/Bar/Blah/Bleep/../../../Biz/";
          std::string canonical = "Foo/Biz/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
        SECTION("last") {
          std::string pathName = "Foo/Bar/Blah/Bleep/../../../";
          std::string canonical = "Foo/";
          p.set(pathName, MonitorElementData::Path::Type::DIR);
          REQUIRE(p.getDirname() == canonical);
          REQUIRE(p.getObjectname().empty());
          REQUIRE(p.getFullname() == canonical);
        }
      }
    }
  }
  SECTION("object") {
    SECTION("no dir") {
      std::string pathName = "bar";
      std::string canonical = "";
      std::string objectName = "bar";
      p.set(pathName, MonitorElementData::Path::Type::DIR_AND_NAME);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname() == objectName);
      REQUIRE(p.getFullname() == canonical + objectName);
    }
    SECTION("1 dir") {
      std::string pathName = "Foo/bar";
      std::string canonical = "Foo/";
      std::string objectName = "bar";
      p.set(pathName, MonitorElementData::Path::Type::DIR_AND_NAME);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname() == objectName);
      REQUIRE(p.getFullname() == canonical + objectName);
    }
    SECTION("2 dir") {
      std::string pathName = "Foo/Biz/bar";
      std::string canonical = "Foo/Biz/";
      std::string objectName = "bar";
      p.set(pathName, MonitorElementData::Path::Type::DIR_AND_NAME);
      REQUIRE(p.getDirname() == canonical);
      REQUIRE(p.getObjectname() == objectName);
      REQUIRE(p.getFullname() == canonical + objectName);
    }
  }
}
