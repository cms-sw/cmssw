#include "RPCReadOutMappingWithFastSearch.h"
#include <vector>
#include <iostream>

using namespace std;

bool RPCReadOutMappingWithFastSearch::lessMap::operator()(
    const LinkBoardElectronicIndex & lb1, const LinkBoardElectronicIndex & lb2) const 
{
  if ( lb1.dccId              < lb2.dccId)              return true;
  if ( lb1.dccId              > lb2.dccId)              return false;
  if ( lb1.dccInputChannelNum < lb2.dccInputChannelNum) return true;
  if ( lb1.dccInputChannelNum > lb2.dccInputChannelNum) return false;
  if ( lb1.tbLinkInputNum     < lb2.tbLinkInputNum)     return true;
  if ( lb1.tbLinkInputNum     > lb2.tbLinkInputNum)     return false;
  if ( lb1.lbNumInLink        < lb2.lbNumInLink)        return true;
  if ( lb1.lbNumInLink        > lb2.lbNumInLink)        return false;
  return false;

}

RPCReadOutMappingWithFastSearch::RPCReadOutMappingWithFastSearch()
   : theMapping(0)
{}

void RPCReadOutMappingWithFastSearch::init(const RPCReadOutMapping * arm)
{
  if (theVersion==arm->version()) return;

  theVersion=arm->version();
  theLBMap.clear();
  theMapping = arm;

  typedef vector<const DccSpec*> DCCLIST;
  DCCLIST dccList = arm->dccList();
  for (DCCLIST::const_iterator idcc = dccList.begin(), idccEnd = dccList.end(); 
      idcc < idccEnd; ++idcc) {
    const DccSpec & dccSpec = **idcc;
    const std::vector<TriggerBoardSpec> & triggerBoards = dccSpec.triggerBoards();
    for ( std::vector<TriggerBoardSpec>::const_iterator
        it = triggerBoards.begin(); it != triggerBoards.end(); it++) {
      const TriggerBoardSpec & triggerBoard = (*it);
      typedef std::vector<const LinkConnSpec* > LINKS;
      LINKS linkConns = triggerBoard.enabledLinkConns();
      for ( LINKS::const_iterator ic = linkConns.begin(); ic != linkConns.end(); ic++) {

        const LinkConnSpec & link = **ic;
        const std::vector<LinkBoardSpec> & boards = link.linkBoards();
        for ( std::vector<LinkBoardSpec>::const_iterator
            ib = boards.begin(); ib != boards.end(); ib++) {

          const LinkBoardSpec & board = (*ib);

          LinkBoardElectronicIndex eleIndex;
          eleIndex.dccId = dccSpec.id();
          eleIndex.dccInputChannelNum = triggerBoard.dccInputChannelNum();
          eleIndex.tbLinkInputNum = link.triggerBoardInputNumber();
          eleIndex.lbNumInLink = board.linkBoardNumInLink();
          LBMap::iterator inMap = theLBMap.find(eleIndex);  
          if (inMap != theLBMap.end()) {
            cout <<"The element in map already exists!"<< endl;
          } else {
            theLBMap[eleIndex] = &board;
          }
        }
      }  
    }
  }
}

RPCReadOutMapping::StripInDetUnit RPCReadOutMappingWithFastSearch::detUnitFrame(
    const LinkBoardSpec& location, const LinkBoardPackedStrip & lbstrip) const
{
  return theMapping->detUnitFrame(location,lbstrip);
}

const LinkBoardSpec* RPCReadOutMappingWithFastSearch::location(const LinkBoardElectronicIndex & ele) const
{
  LBMap::const_iterator inMap = theLBMap.find(ele);
  return (inMap!= theLBMap.end()) ? inMap->second : 0;
// return theMapping->location(ele);
}
