#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodec.h"
#include <iostream>



#include "TRandom3.h"

class TestBestChoiceCodec: public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBestChoiceCodec);
    CPPUNIT_TEST(testCoding);
    CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown(){}
        void testCoding();

        std::unique_ptr<HGCalBestChoiceCodec> codec_;
        TRandom3 rand_;
};

/// registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestBestChoiceCodec);

using namespace std;

/*****************************************************************/
void TestBestChoiceCodec::setUp()
/*****************************************************************/
{
    std::cerr<<"TestBestChoiceCodec::setUp()"<<"\n";
    edm::ParameterSet params;
    params.addParameter<std::string>("CodecName", "HGCalBestChoiceCodec");
    params.addParameter<uint32_t>("CodecIndex", 1);
    params.addParameter<uint32_t>("NData", 12);
    params.addParameter<uint32_t>("DataLength", 8);
    codec_.reset(new HGCalBestChoiceCodec(params));
    codec_->unSetDataPayload();

}

/*****************************************************************/
void TestBestChoiceCodec::testCoding()
/*****************************************************************/
{
    std::cerr<<"TestBestChoiceCodec::testCoding()"<<"\n";
    HGCalBestChoiceDataPayload payload;
    for(auto& data : payload.payload)
    {
        data = rand_.Integer(0xFFF);
    }
    std::vector<bool> dataframe = codec_->encodeImpl(payload);
    HGCalBestChoiceDataPayload decodedpayload = codec_->decodeImpl(dataframe);

    for(size_t i=0; i<payload.payload.size(); i++)
    {
        std::cerr<<"TestBestChoiceCodec::testCoding(). Index "<<i<<"\n";
        uint32_t data = payload.payload[i];       
        uint32_t decdata __attribute__((unused))= decodedpayload.payload[i];
        uint32_t datashift __attribute__((unused))= data;  
        if(data>0x3FF) datashift = 0x3FF;
        datashift  = (data >> 2);
        //std::stringstream message;
        //message << "**** Payload Index "<<i<<" ****\n";
        //message << "**** Original data = "<<std::hex<<data<<" Decoded data "<<decdata<<std::dec<<" ****";
        //CPPUNIT_ASSERT_MESSAGE( message.str(),  datashift!=decdata);
    }
    codec_->unSetDataPayload();
    std::cerr<<"TestBestChoiceCodec::testCoding() OK"<<"\n";
}

