

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTag/XMLCalibration/interface/AlgorithmCalibration.h"
#include "RecoBTag/XMLCalibration/interface/CalibratedHistogramXML.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationCategory.h"

#include <iostream>
using namespace std;
class TestCategory;

///This is an example of how to use the AlgorithmCalibration stuff
// to read the calibrated objects from a .xml file

class XMLCalibrationTest : public edm::EDAnalyzer {
public:
  explicit XMLCalibrationTest(const edm::ParameterSet&);
  ~XMLCalibrationTest() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  AlgorithmCalibration<TestCategory, CalibratedHistogramXML>* m_calib;
};

//This class define calibration  "category" in this example
//the category is simply a range of float: i.e. the category match if
//the calibrationInput (which is a float in that case) is min<= input < max
//In the example .xml files I put two categories one with range 0..2.5 the other with
// rang 2.5...5
class TestCategory : public CalibrationCategory<float> {
public:
  bool match(const float& input) const override  // const reference  for float is stupid but input object
                                                 // are not always  floats
  {
    return (input < m_max) && (input >= m_min);
  }
  std::string name() override { return "TestCategory"; }

  void readFromDOM(DOMElement* dom) override {
    m_min = CalibrationXML::readAttribute<float>(dom, "min");
    m_max = CalibrationXML::readAttribute<float>(dom, "max");
  }

  void saveToDOM(DOMElement* dom) override {
    CalibrationXML::writeAttribute(dom, "min", m_min);
    CalibrationXML::writeAttribute(dom, "max", m_max);
  }

protected:
  float m_min;
  float m_max;
};

XMLCalibrationTest::XMLCalibrationTest(const edm::ParameterSet& iConfig) {
  m_calib = new AlgorithmCalibration<TestCategory, CalibratedHistogramXML>("test.xml");
}

XMLCalibrationTest::~XMLCalibrationTest() { delete m_calib; }

void XMLCalibrationTest::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  //ask the algorithm the calibrated object for an input value
  // of 1.2
  // this will search for the matching category and return the associated calibrated
  // object.

  const CalibratedHistogram* histo = m_calib->getCalibData((float)1.2);

  std::cout << "Pointer of the histogram: " << histo << std::endl;
  std::cout << histo->value(2);
  std::cout << " " << histo->integral(2) << std::endl;
}

DEFINE_FWK_MODULE(XMLCalibrationTest);
