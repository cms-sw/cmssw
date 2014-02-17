/** \file EDMtoMEConverter.cc
 *
 *  See header file for description of class
 *
 *  $Date: 2012/10/10 16:01:52 $
 *  $Revision: 1.36 $
 *  \author M. Strang SUNY-Buffalo
 */

#include <cassert>

#include "DQMServices/Components/plugins/EDMtoMEConverter.h"

using namespace lat;

EDMtoMEConverter::EDMtoMEConverter(const edm::ParameterSet & iPSet) :
  verbosity(0), frequency(0),
  runInputTag_(iPSet.getParameter<edm::InputTag>("runInputTag")),
  lumiInputTag_(iPSet.getParameter<edm::InputTag>("lumiInputTag"))
{
  std::string MsgLoggerCat = "EDMtoMEConverter_EDMtoMEConverter";

  // get information from parameter set
  name = iPSet.getUntrackedParameter<std::string>("Name");
  verbosity = iPSet.getUntrackedParameter<int>("Verbosity");
  frequency = iPSet.getUntrackedParameter<int>("Frequency");

  convertOnEndLumi = iPSet.getUntrackedParameter<bool>("convertOnEndLumi",true);
  convertOnEndRun = iPSet.getUntrackedParameter<bool>("convertOnEndRun",true);

  // use value of first digit to determine default output level (inclusive)
  // 0 is none, 1 is basic, 2 is fill output, 3 is gather output
  verbosity %= 10;

  // get dqm info
  dbe = 0;
  dbe = edm::Service<DQMStore>().operator->();

  // print out Parameter Set information being used
  if (verbosity >= 0) {
    edm::LogInfo(MsgLoggerCat)
      << "\n===============================\n"
      << "Initialized as EDAnalyzer with parameter values:\n"
      << "    Name          = " << name << "\n"
      << "    Verbosity     = " << verbosity << "\n"
      << "    Frequency     = " << frequency << "\n"
      << "===============================\n";
  }

  classtypes.clear();
  classtypes.push_back("TH1F");
  classtypes.push_back("TH1S");
  classtypes.push_back("TH1D");
  classtypes.push_back("TH2F");
  classtypes.push_back("TH2S");
  classtypes.push_back("TH2D");
  classtypes.push_back("TH3F");
  classtypes.push_back("TProfile");
  classtypes.push_back("TProfile2D");
  classtypes.push_back("Double");
  classtypes.push_back("Int");
  classtypes.push_back("Int64");
  classtypes.push_back("String");

  iCountf = 0;
  iCount.clear();

  assert(sizeof(int64_t) == sizeof(long long));

} // end constructor

EDMtoMEConverter::~EDMtoMEConverter() {}

void EDMtoMEConverter::beginJob()
{
}

void EDMtoMEConverter::endJob()
{
  std::string MsgLoggerCat = "EDMtoMEConverter_endJob";
  if (verbosity >= 0)
    edm::LogInfo(MsgLoggerCat)
      << "Terminating having processed " << iCount.size() << " runs across "
      << iCountf << " files.";
  return;
}

void EDMtoMEConverter::respondToOpenInputFile(const edm::FileBlock& iFb)
{
  ++iCountf;
  return;
}

void EDMtoMEConverter::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  std::string MsgLoggerCat = "EDMtoMEConverter_beginRun";

  int nrun = iRun.run();

  // keep track of number of unique runs processed
  ++iCount[nrun];

  if (verbosity > 0) {
    edm::LogInfo(MsgLoggerCat)
      << "Processing run " << nrun << " (" << iCount.size() << " runs total)";
  } else if (verbosity == 0) {
    if (nrun%frequency == 0 || iCount.size() == 1) {
      edm::LogInfo(MsgLoggerCat)
        << "Processing run " << nrun << " (" << iCount.size() << " runs total)";
    }
  }

}

void EDMtoMEConverter::endRun(const edm::Run& iRun, const edm::EventSetup& iSetup)
{
  if (convertOnEndRun) {
    getData(iRun, true);
  }
}

void EDMtoMEConverter::beginLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup)
{
}

void EDMtoMEConverter::endLuminosityBlock(const edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup)
{
  if (convertOnEndLumi) {
    getData(iLumi, false);
  }
}

void EDMtoMEConverter::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}

template <class T>
void
EDMtoMEConverter::getData(T& iGetFrom, bool iEndRun)
{
  edm::InputTag* inputTag = 0;
  if (iEndRun) {
    inputTag = &runInputTag_;
  } else {
    inputTag = &lumiInputTag_;
  }

  std::string MsgLoggerCat = "EDMtoMEConverter_getData";

  if (verbosity >= 0)
    edm::LogInfo (MsgLoggerCat) << "\nRestoring MonitorElements.";

  for (unsigned int ii = 0; ii < classtypes.size(); ++ii) {

    if (classtypes[ii] == "TH1F") {
      edm::Handle<MEtoEDM<TH1F> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH1F> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH1F>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me1.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me1[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me1[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me1[i] && me1[i]->getTH1F() && me1[i]->getTH1F()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me1[i]->getTH1F()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me1[i] = dbe->book1D(metoedmobject[i].object.GetName(),
                                 &metoedmobject[i].object);
          }
          if (!iEndRun) me1[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me1[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH1F creation

    if (classtypes[ii] == "TH1S") {
      edm::Handle<MEtoEDM<TH1S> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH1S> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH1S>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me1.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me1[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me1[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me1[i] && me1[i]->getTH1S() && me1[i]->getTH1S()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me1[i]->getTH1S()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me1[i] = dbe->book1S(metoedmobject[i].object.GetName(),
                                 &metoedmobject[i].object);
          }
          if (!iEndRun) me1[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me1[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH1S creation

    if (classtypes[ii] == "TH1D") {
      edm::Handle<MEtoEDM<TH1D> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH1D> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH1D>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me1.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me1[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me1[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me1[i] && me1[i]->getTH1D() && me1[i]->getTH1D()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me1[i]->getTH1D()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me1[i] = dbe->book1DD(metoedmobject[i].object.GetName(),
                                  &metoedmobject[i].object);
          }
          if (!iEndRun) me1[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me1[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH1D creation

    if (classtypes[ii] == "TH2F") {
      edm::Handle<MEtoEDM<TH2F> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH2F> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH2F>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me2.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me2[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me2[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me2[i] && me2[i]->getTH2F() && me2[i]->getTH2F()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me2[i]->getTH2F()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me2[i] = dbe->book2D(metoedmobject[i].object.GetName(),
                                 &metoedmobject[i].object);
          }
          if (!iEndRun) me2[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me2[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH2F creation

    if (classtypes[ii] == "TH2S") {
      edm::Handle<MEtoEDM<TH2S> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH2S> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH2S>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me2.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me2[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me2[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me2[i] && me2[i]->getTH2S() && me2[i]->getTH2S()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me2[i]->getTH2S()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me2[i] = dbe->book2S(metoedmobject[i].object.GetName(),
                                 &metoedmobject[i].object);
          }
          if (!iEndRun) me2[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me2[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH2S creation

    if (classtypes[ii] == "TH2D") {
      edm::Handle<MEtoEDM<TH2D> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH2D> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH2D>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me2.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me2[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me2[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me2[i] && me2[i]->getTH2D() && me2[i]->getTH2D()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me2[i]->getTH2D()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me2[i] = dbe->book2DD(metoedmobject[i].object.GetName(),
                                  &metoedmobject[i].object);
          }
          if (!iEndRun) me2[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me2[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH2D creation

    if (classtypes[ii] == "TH3F") {
      edm::Handle<MEtoEDM<TH3F> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TH3F> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TH3F>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me3.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me3[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me3[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me3[i] && me3[i]->getTH3F() && me3[i]->getTH3F()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me3[i]->getTH3F()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me3[i] = dbe->book3D(metoedmobject[i].object.GetName(),
                                 &metoedmobject[i].object);
          }
          if (!iEndRun) me3[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me3[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TH3F creation

    if (classtypes[ii] == "TProfile") {
      edm::Handle<MEtoEDM<TProfile> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TProfile> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TProfile>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me4.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me4[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

	std::string name = metoedmobject[i].object.GetName();
        // define new monitor element
        if (dbe) {
          me4[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me4[i] && me4[i]->getTProfile() && me4[i]->getTProfile()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me4[i]->getTProfile()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                      << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me4[i] = dbe->bookProfile(metoedmobject[i].object.GetName(),
                                    &metoedmobject[i].object);
          }
          if (!iEndRun) me4[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me4[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TProfile creation

    if (classtypes[ii] == "TProfile2D") {
      edm::Handle<MEtoEDM<TProfile2D> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TProfile2D> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TProfile2D>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me5.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me5[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          me5[i] = dbe->get(dir+"/"+metoedmobject[i].object.GetName());
          if (me5[i] && me5[i]->getTProfile2D() && me5[i]->getTProfile2D()->TestBit(TH1::kCanRebin) == true) {
            TList list;
            list.Add(&metoedmobject[i].object);
            if (me5[i]->getTProfile2D()->Merge(&list) == -1)
              std::cout << "ERROR EDMtoMEConverter::getData(): merge failed for '"
                        << metoedmobject[i].object.GetName() << "'" <<  std::endl;
          } else {
            dbe->setCurrentFolder(dir);
            me5[i] = dbe->bookProfile2D(metoedmobject[i].object.GetName(),
                                        &metoedmobject[i].object);
          }
          if (!iEndRun) me5[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me5[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end TProfile2D creation

    if (classtypes[ii] == "Double") {
      edm::Handle<MEtoEDM<double> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<double> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<double>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me6.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me6[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;
        std::string name;

        // deconstruct path from fullpath

        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          me6[i] = dbe->bookFloat(name);
          me6[i]->Fill(metoedmobject[i].object);
          if (!iEndRun) me6[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me6[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end Float creation

    if (classtypes[ii] == "Int64") {
      edm::Handle<MEtoEDM<long long> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<long long> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<long long>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me7.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me7[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;
        std::string name;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          long long ival = 0;
          if ( iEndRun ) {
            if (name.find("processedEvents") != std::string::npos) {
              if (MonitorElement* me = dbe->get(dir+"/"+name)) {
                ival = me->getIntValue();
              }
            }
          }
          me7[i] = dbe->bookInt(name);
          me7[i]->Fill(metoedmobject[i].object+ival);
          if (!iEndRun) me7[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me7[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end Int creation

    if (classtypes[ii] == "Int") {
      edm::Handle<MEtoEDM<int> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<int> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<int>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me7.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me7[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;
        std::string name;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          int ival = 0;
          if ( iEndRun ) {
            if (name.find("processedEvents") != std::string::npos) {
              if (MonitorElement* me = dbe->get(dir+"/"+name)) {
                ival = me->getIntValue();
              }
            }
          }
          me7[i] = dbe->bookInt(name);
          me7[i]->Fill(metoedmobject[i].object+ival);
          if (!iEndRun) me7[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me7[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end Int creation

    if (classtypes[ii] == "String") {
      edm::Handle<MEtoEDM<TString> > metoedm;
      iGetFrom.getByLabel(*inputTag, metoedm);

      if (!metoedm.isValid()) {
        //edm::LogWarning(MsgLoggerCat)
        //  << "MEtoEDM<TString> doesn't exist in run";
        continue;
      }

      std::vector<MEtoEDM<TString>::MEtoEDMObject> metoedmobject =
        metoedm->getMEtoEdmObject();

      me8.resize(metoedmobject.size());

      for (unsigned int i = 0; i < metoedmobject.size(); ++i) {

        me8[i] = 0;

        // get full path of monitor element
        std::string pathname = metoedmobject[i].name;
        if (verbosity > 0) std::cout << pathname << std::endl;

        std::string dir;
        std::string name;

        // deconstruct path from fullpath
        StringList fulldir = StringOps::split(pathname,"/");
        name = *(fulldir.end() - 1);

        for (unsigned j = 0; j < fulldir.size() - 1; ++j) {
          dir += fulldir[j];
          if (j != fulldir.size() - 2) dir += "/";
        }

        // define new monitor element
        if (dbe) {
          dbe->setCurrentFolder(dir);
          std::string scont = metoedmobject[i].object.Data();
          me8[i] = dbe->bookString(name,scont);
          if (!iEndRun) me8[i]->setLumiFlag();
        } // end define new monitor elements

        // attach taglist
        TagList tags = metoedmobject[i].tags;

        for (unsigned int j = 0; j < tags.size(); ++j) {
          dbe->tag(me8[i]->getFullname(),tags[j]);
        }
      } // end loop thorugh metoedmobject
    } // end String creation
  }

  // verify tags stored properly
  if (verbosity > 0) {
    std::vector<std::string> stags;
    dbe->getAllTags(stags);
    for (unsigned int i = 0; i < stags.size(); ++i) {
      std::cout << "Tags: " << stags[i] << std::endl;
    }
  }

}

