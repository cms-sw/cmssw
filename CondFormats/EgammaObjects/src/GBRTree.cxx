#include "CondFormats/EgammaObjects/interface/GBRTree.h"

//_______________________________________________________________________
GBRTree::GBRTree() {}

//_______________________________________________________________________
GBRTree::GBRTree(int nIntermediate, int nTerminal)
{
  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  
  fCutIndices.reserve(nIntermediate);
  fCutVals.reserve(nIntermediate);
  fLeftIndices.reserve(nIntermediate);
  fRightIndices.reserve(nIntermediate);
  fResponses.reserve(nTerminal);
}

//_______________________________________________________________________
void GBRTree::addIntermediateNode(unsigned char cutIndex, float cutVal, bool leftIsTerminal, bool rightIsTerminal)
{
    int thisidx = fCutIndices.size();
    fCutIndices.push_back(cutIndex);

    fCutVals.push_back(cutVal);

    fLeftIndices.push_back(0);
    fRightIndices.push_back(0);

    fLeftIndices[thisidx]  = leftIsTerminal  ? -fResponses.size() : fCutIndices.size();
    fRightIndices[thisidx] = rightIsTerminal ? -fResponses.size() : fCutIndices.size();
}

//_______________________________________________________________________
void GBRTree::addTerminalNode(float response)
{
    fResponses.push_back(response);

    //special case, root node is terminal, create fake intermediate node at root
    if (fCutIndices.empty()) {
      fCutIndices.push_back(0);
      fCutVals.push_back(0);
      fLeftIndices.push_back(0);
      fRightIndices.push_back(0);
    }

}
