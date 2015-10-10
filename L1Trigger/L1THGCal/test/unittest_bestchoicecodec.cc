#include <cppunit/extensions/HelperMacros.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/fe_codecs/HGCalBestChoiceCodecImpl.h"
#include <iostream>



#include "TRandom3.h"

class TestBestChoiceCodec: public CppUnit::TestFixture
{
    CPPUNIT_TEST_SUITE(TestBestChoiceCodec);
    CPPUNIT_TEST(testEncodingDecodingConsistency);
    CPPUNIT_TEST_SUITE_END();

    public:
        void setUp();
        void tearDown(){}
        void testEncodingDecodingConsistency();

        std::unique_ptr<HGCalBestChoiceCodecImpl> codec_;
        TRandom3 rand_;
};

/// registration of the test so that the runner can find it
CPPUNIT_TEST_SUITE_REGISTRATION(TestBestChoiceCodec);

/*****************************************************************/
void TestBestChoiceCodec::setUp()
/*****************************************************************/
{
    edm::ParameterSet params;
    params.addParameter<std::string>("CodecName", "HGCalBestChoiceCodec");
    params.addParameter<uint32_t>("CodecIndex", 1);
    params.addParameter<uint32_t>("NData", 12);
    params.addParameter<uint32_t>("DataLength", 8);
    codec_.reset(new HGCalBestChoiceCodecImpl(params));

}

/*****************************************************************/
void TestBestChoiceCodec::testEncodingDecodingConsistency()
/*****************************************************************/
{
    HGCalBestChoiceDataPayload payload;
    for(unsigned round=0; round<1000; round++)
    {
        payload.reset();
        for(auto& data : payload.payload)
        {
            data = rand_.Integer(0x200);
        }
        codec_->bestChoiceSelect(payload);
        std::vector<bool> dataframe = codec_->encode(payload);
        HGCalBestChoiceDataPayload decodedpayload = codec_->decode(dataframe);

        for(size_t i=0; i<payload.payload.size(); i++)
        {
            uint32_t data = payload.payload[i];       
            uint32_t decdata = decodedpayload.payload[i];
            uint32_t datashift = data;  
            if(datashift>0x3FF) datashift = 0x3FF;
            datashift  = (datashift >> 2);
            std::stringstream message;
            message << "\n**** Payload Index "<<i<<" ****\n";
            message << "**** Original 12 bit data = 0x"<<std::hex<<data<<" ****\n";
            message << "**** Original  8 bit data = 0x"<<datashift<<" ****\n";
            message << "**** Decoded  data = 0x"<<decdata<<std::dec<<" ****\n";
            CPPUNIT_ASSERT_MESSAGE( message.str(),  datashift==decdata);
        }
    }
}

