#include "HtrXmlPatternWriter.h"

static const char* tabbing(int level) {
  static char tab[50];
  for (int i=0; i<level*2; i++)
    tab[i]=' ';
  tab[level*2]=0;
  return tab;
}

HtrXmlPatternWriter::HtrXmlPatternWriter() {
  // set the timestamp!
  time_t now1=time(0);
  struct tm* now=localtime(&now1);

  char buffer[1024];
  strftime(buffer,1024,"%Y-%m-%d %H:%M:%S",now);

  m_stamp=buffer;
}

void HtrXmlPatternWriter::writePattern(HalfHtrData* spigotData, int fiber, std::ostream& os, int level) {

  os << tabbing(level) << "<CFGBrick>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='DCC' type='int'>" << std::dec << spigotData->getDCC() << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='SPIGOT' type='int'>" << std::dec << spigotData->getSpigot() << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='CRATE' type='int'>" << std::dec << spigotData->getCrate() << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='SLOT' type='int'>" << spigotData->getSlot() << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='TOPBOTTOM' type='int'>" << spigotData->getTB() << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='FIBER' type='int'>" << fiber << "</Parameter>" << std::endl;
  
  int genIndex=fiber+(spigotData->getTB()*10)+(spigotData->getSlot()*100)+(spigotData->getCrate()*10000);
  os << tabbing(level+1) << "<Parameter name='GENERALIZEDINDEX' type='int'>" << std::dec << genIndex << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='CREATIONTAG' type='string'>" << m_tagName << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='CREATIONSTAMP' type='string'>" << m_stamp << "</Parameter>" << std::endl;
  os << tabbing(level+1) << "<Parameter name='PATTERN_SPEC_NAME' type='string'>" << m_tagName << "</Parameter>" << std::endl;
  

  // CREATIONTAG, CREATIONSTAMP, PATTERN_SPEC_NAME
  
  os << tabbing(level+1) << "<Data elements='1024' encoding='hex'>" << std::endl;
  
  ChannelPattern* p1=spigotData->getPattern((fiber-1)*3+1);
  ChannelPattern* p2=spigotData->getPattern((fiber-1)*3+2);
  ChannelPattern* p3=spigotData->getPattern((fiber-1)*3+3);


  unsigned int w1,w2;
  for (int i=0; i<512; i++) {
    packWordsStd(p1->getCoded(i),p2->getCoded(i),p3->getCoded(i),i%4,w1,w2);
    os << std::hex << w1 << ' ' << std::hex << w2 << ' ';
  }

  os << std::endl << tabbing(level+1) << "</Data>" << std::endl;
  
  os << tabbing(level) << "</CFGBrick>" << std::endl;
}

void HtrXmlPatternWriter::packWordsStd(int adc0, int adc1, int adc2, int capid, unsigned int& w1, unsigned int& w2) {
  w1=0x3;
  w1|=(capid&0x3)<<3;
  w1|=(capid&0x3)<<5;
  w1|=(capid&0x3)<<7;
  w1|=adc2<<9;
  w1|=0x10000; // data valid
  w2=adc1<<1;
  w2|=adc0<<9;
  w2|=0x10000; // data valid
}
  


