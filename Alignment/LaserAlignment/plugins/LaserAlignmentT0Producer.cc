// -*- C++ -*-
//
// Package:    LaserAlignmentT0Producer
// Class:      LaserAlignmentT0Producer
//

#include "Alignment/LaserAlignment/plugins/LaserAlignmentT0Producer.h"

///
/// Loops all input SiStripDigi or SiStripRawDigi collections
/// and announces the corresponding product
///
LaserAlignmentT0Producer::LaserAlignmentT0Producer(const edm::ParameterSet& iConfig) {
  // alias for the Branches in the output file
  std::string alias(iConfig.getParameter<std::string>("@module_label"));

  // the list of input digi products from the cfg
  digiProducerList = iConfig.getParameter<std::vector<edm::ParameterSet>>("DigiProducerList");

  // loop all input products
  for (std::vector<edm::ParameterSet>::iterator aDigiProducer = digiProducerList.begin();
       aDigiProducer != digiProducerList.end();
       ++aDigiProducer) {
    std::string digiProducer = aDigiProducer->getParameter<std::string>("DigiProducer");
    std::string digiLabel = aDigiProducer->getParameter<std::string>("DigiLabel");
    std::string digiType = aDigiProducer->getParameter<std::string>("DigiType");

    // check if raw/processed digis in this collection
    if (digiType == "Raw") {
      produces<edm::DetSetVector<SiStripRawDigi>>(digiLabel).setBranchAlias(alias + "siStripRawDigis");
    } else if (digiType == "Processed") {  // "ZeroSuppressed" digis (non-raw)
      produces<edm::DetSetVector<SiStripDigi>>(digiLabel).setBranchAlias(alias + "siStripDigis");
    } else {
      throw cms::Exception("LaserAlignmentT0Producer")
          << "ERROR ** Unknown DigiType: " << digiType << " specified in cfg file" << std::endl;
    }
  }
}

///
///
///
LaserAlignmentT0Producer::~LaserAlignmentT0Producer() {}

///
/// outline:
/// * loops alls input strip digi products
/// * branches depending on if it contains SiStripDigi or SiStripRawDigi
/// * for each product: selects only LAS module DetSets
///   and copies them to new DetSetVector
///
void LaserAlignmentT0Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;

  // loop all input products
  for (std::vector<edm::ParameterSet>::iterator aDigiProducer = digiProducerList.begin();
       aDigiProducer != digiProducerList.end();
       ++aDigiProducer) {
    std::string digiProducer = aDigiProducer->getParameter<std::string>("DigiProducer");
    std::string digiLabel = aDigiProducer->getParameter<std::string>("DigiLabel");
    std::string digiType = aDigiProducer->getParameter<std::string>("DigiType");

    // now a distinction of cases: raw or processed digis?

    // first we go for raw digis => SiStripRawDigi
    if (digiType == "Raw") {
      // retrieve the SiStripRawDigis collection
      edm::Handle<edm::DetSetVector<SiStripRawDigi>> theStripDigis;
      iEvent.getByLabel(digiProducer, digiLabel, theStripDigis);

      // empty container
      std::vector<edm::DetSet<SiStripRawDigi>> theDigiVector;

      // loop the incoming DetSetVector
      for (edm::DetSetVector<SiStripRawDigi>::const_iterator aDetSet = theStripDigis->begin();
           aDetSet != theStripDigis->end();
           ++aDetSet) {
        // is the current DetSet that of a LAS module?
        if (find(theLasDetIds.begin(), theLasDetIds.end(), aDetSet->detId()) != theLasDetIds.end()) {
          // yes it's a LAS module, so copy the Digis to the output

          // a DetSet for output
          edm::DetSet<SiStripRawDigi> outputDetSet(aDetSet->detId());

          // copy all the digis to the output DetSet. Unfortunately, there's no copy constructor..
          for (edm::DetSet<SiStripRawDigi>::const_iterator aDigi = aDetSet->begin(); aDigi != aDetSet->end(); ++aDigi) {
            outputDetSet.push_back(*aDigi);
          }

          // onto the later DetSetVector
          theDigiVector.push_back(outputDetSet);
        }
      }

      // create the output collection for the DetSetVector
      // write output to file
      iEvent.put(std::make_unique<edm::DetSetVector<SiStripRawDigi>>(theDigiVector), digiLabel);

    }

    // then we assume "ZeroSuppressed" (non-raw) => SiStripDigi
    // and do exactly the same as above
    else if (digiType == "Processed") {
      edm::Handle<edm::DetSetVector<SiStripDigi>> theStripDigis;
      iEvent.getByLabel(digiProducer, digiLabel, theStripDigis);

      std::vector<edm::DetSet<SiStripDigi>> theDigiVector;

      for (edm::DetSetVector<SiStripDigi>::const_iterator aDetSet = theStripDigis->begin();
           aDetSet != theStripDigis->end();
           ++aDetSet) {
        if (find(theLasDetIds.begin(), theLasDetIds.end(), aDetSet->detId()) != theLasDetIds.end()) {
          edm::DetSet<SiStripDigi> outputDetSet(aDetSet->detId());
          for (edm::DetSet<SiStripDigi>::const_iterator aDigi = aDetSet->begin(); aDigi != aDetSet->end(); ++aDigi) {
            outputDetSet.push_back(*aDigi);
          }

          theDigiVector.push_back(outputDetSet);
        }
      }

      iEvent.put(std::make_unique<edm::DetSetVector<SiStripDigi>>(theDigiVector), digiLabel);

    }

    else {
      throw cms::Exception("LaserAlignmentT0Producer")
          << "ERROR ** Unknown DigiType: " << digiType << " specified in cfg file" << std::endl;
    }

  }  // loop all input products
}

///
///
///
void LaserAlignmentT0Producer::beginJob() {
  // fill the vector with LAS det ids
  FillDetIds();
}

///
///
///
void LaserAlignmentT0Producer::endJob() {}

///
/// fill a vector with all the det ids of the 434
/// siStrip modules relevant to the LAS
///
/// (ugly mechanism, should later be
/// replaced by some hash-based)
///
void LaserAlignmentT0Producer::FillDetIds(void) {
  theLasDetIds.resize(0);

  // TEC+ internal alignment modules
  theLasDetIds.push_back(470307208);
  theLasDetIds.push_back(470323592);
  theLasDetIds.push_back(470339976);
  theLasDetIds.push_back(470356360);
  theLasDetIds.push_back(470372744);
  theLasDetIds.push_back(470389128);
  theLasDetIds.push_back(470405512);
  theLasDetIds.push_back(470421896);
  theLasDetIds.push_back(470438280);
  theLasDetIds.push_back(470307464);
  theLasDetIds.push_back(470323848);
  theLasDetIds.push_back(470340232);
  theLasDetIds.push_back(470356616);
  theLasDetIds.push_back(470373000);
  theLasDetIds.push_back(470389384);
  theLasDetIds.push_back(470405768);
  theLasDetIds.push_back(470422152);
  theLasDetIds.push_back(470438536);
  theLasDetIds.push_back(470307720);
  theLasDetIds.push_back(470324104);
  theLasDetIds.push_back(470340488);
  theLasDetIds.push_back(470356872);
  theLasDetIds.push_back(470373256);
  theLasDetIds.push_back(470389640);
  theLasDetIds.push_back(470406024);
  theLasDetIds.push_back(470422408);
  theLasDetIds.push_back(470438792);
  theLasDetIds.push_back(470307976);
  theLasDetIds.push_back(470324360);
  theLasDetIds.push_back(470340744);
  theLasDetIds.push_back(470357128);
  theLasDetIds.push_back(470373512);
  theLasDetIds.push_back(470389896);
  theLasDetIds.push_back(470406280);
  theLasDetIds.push_back(470422664);
  theLasDetIds.push_back(470439048);
  theLasDetIds.push_back(470308232);
  theLasDetIds.push_back(470324616);
  theLasDetIds.push_back(470341000);
  theLasDetIds.push_back(470357384);
  theLasDetIds.push_back(470373768);
  theLasDetIds.push_back(470390152);
  theLasDetIds.push_back(470406536);
  theLasDetIds.push_back(470422920);
  theLasDetIds.push_back(470439304);
  theLasDetIds.push_back(470308488);
  theLasDetIds.push_back(470324872);
  theLasDetIds.push_back(470341256);
  theLasDetIds.push_back(470357640);
  theLasDetIds.push_back(470374024);
  theLasDetIds.push_back(470390408);
  theLasDetIds.push_back(470406792);
  theLasDetIds.push_back(470423176);
  theLasDetIds.push_back(470439560);
  theLasDetIds.push_back(470308744);
  theLasDetIds.push_back(470325128);
  theLasDetIds.push_back(470341512);
  theLasDetIds.push_back(470357896);
  theLasDetIds.push_back(470374280);
  theLasDetIds.push_back(470390664);
  theLasDetIds.push_back(470407048);
  theLasDetIds.push_back(470423432);
  theLasDetIds.push_back(470439816);
  theLasDetIds.push_back(470309000);
  theLasDetIds.push_back(470325384);
  theLasDetIds.push_back(470341768);
  theLasDetIds.push_back(470358152);
  theLasDetIds.push_back(470374536);
  theLasDetIds.push_back(470390920);
  theLasDetIds.push_back(470407304);
  theLasDetIds.push_back(470423688);
  theLasDetIds.push_back(470440072);
  theLasDetIds.push_back(470307272);
  theLasDetIds.push_back(470323656);
  theLasDetIds.push_back(470340040);
  theLasDetIds.push_back(470356424);
  theLasDetIds.push_back(470372808);
  theLasDetIds.push_back(470389192);
  theLasDetIds.push_back(470405576);
  theLasDetIds.push_back(470421960);
  theLasDetIds.push_back(470438344);
  theLasDetIds.push_back(470307528);
  theLasDetIds.push_back(470323912);
  theLasDetIds.push_back(470340296);
  theLasDetIds.push_back(470356680);
  theLasDetIds.push_back(470373064);
  theLasDetIds.push_back(470389448);
  theLasDetIds.push_back(470405832);
  theLasDetIds.push_back(470422216);
  theLasDetIds.push_back(470438600);
  theLasDetIds.push_back(470307784);
  theLasDetIds.push_back(470324168);
  theLasDetIds.push_back(470340552);
  theLasDetIds.push_back(470356936);
  theLasDetIds.push_back(470373320);
  theLasDetIds.push_back(470389704);
  theLasDetIds.push_back(470406088);
  theLasDetIds.push_back(470422472);
  theLasDetIds.push_back(470438856);
  theLasDetIds.push_back(470308040);
  theLasDetIds.push_back(470324424);
  theLasDetIds.push_back(470340808);
  theLasDetIds.push_back(470357192);
  theLasDetIds.push_back(470373576);
  theLasDetIds.push_back(470389960);
  theLasDetIds.push_back(470406344);
  theLasDetIds.push_back(470422728);
  theLasDetIds.push_back(470439112);
  theLasDetIds.push_back(470308296);
  theLasDetIds.push_back(470324680);
  theLasDetIds.push_back(470341064);
  theLasDetIds.push_back(470357448);
  theLasDetIds.push_back(470373832);
  theLasDetIds.push_back(470390216);
  theLasDetIds.push_back(470406600);
  theLasDetIds.push_back(470422984);
  theLasDetIds.push_back(470439368);
  theLasDetIds.push_back(470308552);
  theLasDetIds.push_back(470324936);
  theLasDetIds.push_back(470341320);
  theLasDetIds.push_back(470357704);
  theLasDetIds.push_back(470374088);
  theLasDetIds.push_back(470390472);
  theLasDetIds.push_back(470406856);
  theLasDetIds.push_back(470423240);
  theLasDetIds.push_back(470439624);
  theLasDetIds.push_back(470308808);
  theLasDetIds.push_back(470325192);
  theLasDetIds.push_back(470341576);
  theLasDetIds.push_back(470357960);
  theLasDetIds.push_back(470374344);
  theLasDetIds.push_back(470390728);
  theLasDetIds.push_back(470407112);
  theLasDetIds.push_back(470423496);
  theLasDetIds.push_back(470439880);
  theLasDetIds.push_back(470309064);
  theLasDetIds.push_back(470325448);
  theLasDetIds.push_back(470341832);
  theLasDetIds.push_back(470358216);
  theLasDetIds.push_back(470374600);
  theLasDetIds.push_back(470390984);
  theLasDetIds.push_back(470407368);
  theLasDetIds.push_back(470423752);
  theLasDetIds.push_back(470440136);

  // TEC- internal alignment modules
  theLasDetIds.push_back(470045064);
  theLasDetIds.push_back(470061448);
  theLasDetIds.push_back(470077832);
  theLasDetIds.push_back(470094216);
  theLasDetIds.push_back(470110600);
  theLasDetIds.push_back(470126984);
  theLasDetIds.push_back(470143368);
  theLasDetIds.push_back(470159752);
  theLasDetIds.push_back(470176136);
  theLasDetIds.push_back(470045320);
  theLasDetIds.push_back(470061704);
  theLasDetIds.push_back(470078088);
  theLasDetIds.push_back(470094472);
  theLasDetIds.push_back(470110856);
  theLasDetIds.push_back(470127240);
  theLasDetIds.push_back(470143624);
  theLasDetIds.push_back(470160008);
  theLasDetIds.push_back(470176392);
  theLasDetIds.push_back(470045576);
  theLasDetIds.push_back(470061960);
  theLasDetIds.push_back(470078344);
  theLasDetIds.push_back(470094728);
  theLasDetIds.push_back(470111112);
  theLasDetIds.push_back(470127496);
  theLasDetIds.push_back(470143880);
  theLasDetIds.push_back(470160264);
  theLasDetIds.push_back(470176648);
  theLasDetIds.push_back(470045832);
  theLasDetIds.push_back(470062216);
  theLasDetIds.push_back(470078600);
  theLasDetIds.push_back(470094984);
  theLasDetIds.push_back(470111368);
  theLasDetIds.push_back(470127752);
  theLasDetIds.push_back(470144136);
  theLasDetIds.push_back(470160520);
  theLasDetIds.push_back(470176904);
  theLasDetIds.push_back(470046088);
  theLasDetIds.push_back(470062472);
  theLasDetIds.push_back(470078856);
  theLasDetIds.push_back(470095240);
  theLasDetIds.push_back(470111624);
  theLasDetIds.push_back(470128008);
  theLasDetIds.push_back(470144392);
  theLasDetIds.push_back(470160776);
  theLasDetIds.push_back(470177160);
  theLasDetIds.push_back(470046344);
  theLasDetIds.push_back(470062728);
  theLasDetIds.push_back(470079112);
  theLasDetIds.push_back(470095496);
  theLasDetIds.push_back(470111880);
  theLasDetIds.push_back(470128264);
  theLasDetIds.push_back(470144648);
  theLasDetIds.push_back(470161032);
  theLasDetIds.push_back(470177416);
  theLasDetIds.push_back(470046600);
  theLasDetIds.push_back(470062984);
  theLasDetIds.push_back(470079368);
  theLasDetIds.push_back(470095752);
  theLasDetIds.push_back(470112136);
  theLasDetIds.push_back(470128520);
  theLasDetIds.push_back(470144904);
  theLasDetIds.push_back(470161288);
  theLasDetIds.push_back(470177672);
  theLasDetIds.push_back(470046856);
  theLasDetIds.push_back(470063240);
  theLasDetIds.push_back(470079624);
  theLasDetIds.push_back(470096008);
  theLasDetIds.push_back(470112392);
  theLasDetIds.push_back(470128776);
  theLasDetIds.push_back(470145160);
  theLasDetIds.push_back(470161544);
  theLasDetIds.push_back(470177928);
  theLasDetIds.push_back(470045128);
  theLasDetIds.push_back(470061512);
  theLasDetIds.push_back(470077896);
  theLasDetIds.push_back(470094280);
  theLasDetIds.push_back(470110664);
  theLasDetIds.push_back(470127048);
  theLasDetIds.push_back(470143432);
  theLasDetIds.push_back(470159816);
  theLasDetIds.push_back(470176200);
  theLasDetIds.push_back(470045384);
  theLasDetIds.push_back(470061768);
  theLasDetIds.push_back(470078152);
  theLasDetIds.push_back(470094536);
  theLasDetIds.push_back(470110920);
  theLasDetIds.push_back(470127304);
  theLasDetIds.push_back(470143688);
  theLasDetIds.push_back(470160072);
  theLasDetIds.push_back(470176456);
  theLasDetIds.push_back(470045640);
  theLasDetIds.push_back(470062024);
  theLasDetIds.push_back(470078408);
  theLasDetIds.push_back(470094792);
  theLasDetIds.push_back(470111176);
  theLasDetIds.push_back(470127560);
  theLasDetIds.push_back(470143944);
  theLasDetIds.push_back(470160328);
  theLasDetIds.push_back(470176712);
  theLasDetIds.push_back(470045896);
  theLasDetIds.push_back(470062280);
  theLasDetIds.push_back(470078664);
  theLasDetIds.push_back(470095048);
  theLasDetIds.push_back(470111432);
  theLasDetIds.push_back(470127816);
  theLasDetIds.push_back(470144200);
  theLasDetIds.push_back(470160584);
  theLasDetIds.push_back(470176968);
  theLasDetIds.push_back(470046152);
  theLasDetIds.push_back(470062536);
  theLasDetIds.push_back(470078920);
  theLasDetIds.push_back(470095304);
  theLasDetIds.push_back(470111688);
  theLasDetIds.push_back(470128072);
  theLasDetIds.push_back(470144456);
  theLasDetIds.push_back(470160840);
  theLasDetIds.push_back(470177224);
  theLasDetIds.push_back(470046408);
  theLasDetIds.push_back(470062792);
  theLasDetIds.push_back(470079176);
  theLasDetIds.push_back(470095560);
  theLasDetIds.push_back(470111944);
  theLasDetIds.push_back(470128328);
  theLasDetIds.push_back(470144712);
  theLasDetIds.push_back(470161096);
  theLasDetIds.push_back(470177480);
  theLasDetIds.push_back(470046664);
  theLasDetIds.push_back(470063048);
  theLasDetIds.push_back(470079432);
  theLasDetIds.push_back(470095816);
  theLasDetIds.push_back(470112200);
  theLasDetIds.push_back(470128584);
  theLasDetIds.push_back(470144968);
  theLasDetIds.push_back(470161352);
  theLasDetIds.push_back(470177736);
  theLasDetIds.push_back(470046920);
  theLasDetIds.push_back(470063304);
  theLasDetIds.push_back(470079688);
  theLasDetIds.push_back(470096072);
  theLasDetIds.push_back(470112456);
  theLasDetIds.push_back(470128840);
  theLasDetIds.push_back(470145224);
  theLasDetIds.push_back(470161608);
  theLasDetIds.push_back(470177992);

  // TEC+ alignment tube modules
  theLasDetIds.push_back(470307468);
  theLasDetIds.push_back(470323852);
  theLasDetIds.push_back(470340236);
  theLasDetIds.push_back(470356620);
  theLasDetIds.push_back(470373004);
  theLasDetIds.push_back(470307716);
  theLasDetIds.push_back(470324100);
  theLasDetIds.push_back(470340484);
  theLasDetIds.push_back(470356868);
  theLasDetIds.push_back(470373252);
  theLasDetIds.push_back(470308236);
  theLasDetIds.push_back(470324620);
  theLasDetIds.push_back(470341004);
  theLasDetIds.push_back(470357388);
  theLasDetIds.push_back(470373772);
  theLasDetIds.push_back(470308748);
  theLasDetIds.push_back(470325132);
  theLasDetIds.push_back(470341516);
  theLasDetIds.push_back(470357900);
  theLasDetIds.push_back(470374284);
  theLasDetIds.push_back(470308996);
  theLasDetIds.push_back(470325380);
  theLasDetIds.push_back(470341764);
  theLasDetIds.push_back(470358148);
  theLasDetIds.push_back(470374532);

  // TEC- alignment tube modules
  theLasDetIds.push_back(470045316);
  theLasDetIds.push_back(470061700);
  theLasDetIds.push_back(470078084);
  theLasDetIds.push_back(470094468);
  theLasDetIds.push_back(470110852);
  theLasDetIds.push_back(470045580);
  theLasDetIds.push_back(470061964);
  theLasDetIds.push_back(470078348);
  theLasDetIds.push_back(470094732);
  theLasDetIds.push_back(470111116);
  theLasDetIds.push_back(470046084);
  theLasDetIds.push_back(470062468);
  theLasDetIds.push_back(470078852);
  theLasDetIds.push_back(470095236);
  theLasDetIds.push_back(470111620);
  theLasDetIds.push_back(470046596);
  theLasDetIds.push_back(470062980);
  theLasDetIds.push_back(470079364);
  theLasDetIds.push_back(470095748);
  theLasDetIds.push_back(470112132);
  theLasDetIds.push_back(470046860);
  theLasDetIds.push_back(470063244);
  theLasDetIds.push_back(470079628);
  theLasDetIds.push_back(470096012);
  theLasDetIds.push_back(470112396);

  // TIB alignment tube modules
  theLasDetIds.push_back(369174604);
  theLasDetIds.push_back(369174600);
  theLasDetIds.push_back(369174596);
  theLasDetIds.push_back(369170500);
  theLasDetIds.push_back(369170504);
  theLasDetIds.push_back(369170508);
  theLasDetIds.push_back(369174732);
  theLasDetIds.push_back(369174728);
  theLasDetIds.push_back(369174724);
  theLasDetIds.push_back(369170628);
  theLasDetIds.push_back(369170632);
  theLasDetIds.push_back(369170636);
  theLasDetIds.push_back(369174812);
  theLasDetIds.push_back(369174808);
  theLasDetIds.push_back(369174804);
  theLasDetIds.push_back(369170708);
  theLasDetIds.push_back(369170712);
  theLasDetIds.push_back(369170716);
  theLasDetIds.push_back(369174940);
  theLasDetIds.push_back(369174936);
  theLasDetIds.push_back(369174932);
  theLasDetIds.push_back(369170836);
  theLasDetIds.push_back(369170840);
  theLasDetIds.push_back(369170844);
  theLasDetIds.push_back(369175068);
  theLasDetIds.push_back(369175064);
  theLasDetIds.push_back(369175060);
  theLasDetIds.push_back(369170964);
  theLasDetIds.push_back(369170968);
  theLasDetIds.push_back(369170972);
  theLasDetIds.push_back(369175164);
  theLasDetIds.push_back(369175160);
  theLasDetIds.push_back(369175156);
  theLasDetIds.push_back(369171060);
  theLasDetIds.push_back(369171064);
  theLasDetIds.push_back(369171068);
  theLasDetIds.push_back(369175292);
  theLasDetIds.push_back(369175288);
  theLasDetIds.push_back(369175284);
  theLasDetIds.push_back(369171188);
  theLasDetIds.push_back(369171192);
  theLasDetIds.push_back(369171196);
  theLasDetIds.push_back(369175372);
  theLasDetIds.push_back(369175368);
  theLasDetIds.push_back(369175364);
  theLasDetIds.push_back(369171268);
  theLasDetIds.push_back(369171272);
  theLasDetIds.push_back(369171276);

  // TOB alignment tube modules
  theLasDetIds.push_back(436232314);
  theLasDetIds.push_back(436232306);
  theLasDetIds.push_back(436232298);
  theLasDetIds.push_back(436228198);
  theLasDetIds.push_back(436228206);
  theLasDetIds.push_back(436228214);
  theLasDetIds.push_back(436232506);
  theLasDetIds.push_back(436232498);
  theLasDetIds.push_back(436232490);
  theLasDetIds.push_back(436228390);
  theLasDetIds.push_back(436228398);
  theLasDetIds.push_back(436228406);
  theLasDetIds.push_back(436232634);
  theLasDetIds.push_back(436232626);
  theLasDetIds.push_back(436232618);
  theLasDetIds.push_back(436228518);
  theLasDetIds.push_back(436228526);
  theLasDetIds.push_back(436228534);
  theLasDetIds.push_back(436232826);
  theLasDetIds.push_back(436232818);
  theLasDetIds.push_back(436232810);
  theLasDetIds.push_back(436228710);
  theLasDetIds.push_back(436228718);
  theLasDetIds.push_back(436228726);
  theLasDetIds.push_back(436233018);
  theLasDetIds.push_back(436233010);
  theLasDetIds.push_back(436233002);
  theLasDetIds.push_back(436228902);
  theLasDetIds.push_back(436228910);
  theLasDetIds.push_back(436228918);
  theLasDetIds.push_back(436233146);
  theLasDetIds.push_back(436233138);
  theLasDetIds.push_back(436233130);
  theLasDetIds.push_back(436229030);
  theLasDetIds.push_back(436229038);
  theLasDetIds.push_back(436229046);
  theLasDetIds.push_back(436233338);
  theLasDetIds.push_back(436233330);
  theLasDetIds.push_back(436233322);
  theLasDetIds.push_back(436229222);
  theLasDetIds.push_back(436229230);
  theLasDetIds.push_back(436229238);
  theLasDetIds.push_back(436233466);
  theLasDetIds.push_back(436233458);
  theLasDetIds.push_back(436233450);
  theLasDetIds.push_back(436229350);
  theLasDetIds.push_back(436229358);
  theLasDetIds.push_back(436229366);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LaserAlignmentT0Producer);
