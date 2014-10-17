#include "CondFormats/L1TObjects/interface/L1GtDefinitions.h"
#include <cassert>

int
main(int argc, char **)
{
  assert(l1GtBoardTypeStringToEnum("GTFE") == GTFE);
  assert(l1GtBoardTypeStringToEnum("FDL") == FDL);
  assert(l1GtBoardTypeStringToEnum("PSB") == PSB);
  assert(l1GtBoardTypeStringToEnum("GMT") == GMT);
  assert(l1GtBoardTypeStringToEnum("TCS") == TCS);
  assert(l1GtBoardTypeStringToEnum("TIM") == TIM);
  assert(l1GtBoardTypeStringToEnum("BoardNull") == BoardNull);
  assert(l1GtBoardTypeStringToEnum("FooBar") == BoardNull);

  assert(l1GtBoardTypeEnumToString(GTFE) == "GTFE");
  assert(l1GtBoardTypeEnumToString(FDL) == "FDL");
  assert(l1GtBoardTypeEnumToString(PSB) == "PSB");
  assert(l1GtBoardTypeEnumToString(GMT) == "GMT");
  assert(l1GtBoardTypeEnumToString(TCS) == "TCS");
  assert(l1GtBoardTypeEnumToString(TIM) == "TIM");
  assert(l1GtBoardTypeEnumToString(BoardNull) == "BoardNull");
  assert(l1GtBoardTypeEnumToString((L1GtBoardType)-1) == "BoardNull");

  assert(l1GtPsbQuadStringToEnum("Free") == Free);
  assert(l1GtPsbQuadStringToEnum("TechTr") == TechTr);
  assert(l1GtPsbQuadStringToEnum("IsoEGQ") == IsoEGQ);
  assert(l1GtPsbQuadStringToEnum("NoIsoEGQ") == NoIsoEGQ);
  assert(l1GtPsbQuadStringToEnum("CenJetQ") == CenJetQ);
  assert(l1GtPsbQuadStringToEnum("ForJetQ") == ForJetQ);
  assert(l1GtPsbQuadStringToEnum("TauJetQ") == TauJetQ);
  assert(l1GtPsbQuadStringToEnum("ESumsQ") == ESumsQ);
  assert(l1GtPsbQuadStringToEnum("JetCountsQ") == JetCountsQ);
  assert(l1GtPsbQuadStringToEnum("MQB1") == MQB1);
  assert(l1GtPsbQuadStringToEnum("MQB2") == MQB2);
  assert(l1GtPsbQuadStringToEnum("MQF3") == MQF3);
  assert(l1GtPsbQuadStringToEnum("MQF4") == MQF4);
  assert(l1GtPsbQuadStringToEnum("MQB5") == MQB5);
  assert(l1GtPsbQuadStringToEnum("MQB6") == MQB6);
  assert(l1GtPsbQuadStringToEnum("MQF7") == MQF7);
  assert(l1GtPsbQuadStringToEnum("MQF8") == MQF8);
  assert(l1GtPsbQuadStringToEnum("MQB9") == MQB9);
  assert(l1GtPsbQuadStringToEnum("MQB10") == MQB10);
  assert(l1GtPsbQuadStringToEnum("MQF11") == MQF11);
  assert(l1GtPsbQuadStringToEnum("MQF12") == MQF12);
  assert(l1GtPsbQuadStringToEnum("CastorQ") == CastorQ);
  assert(l1GtPsbQuadStringToEnum("HfQ") == HfQ);
  assert(l1GtPsbQuadStringToEnum("BptxQ") == BptxQ);
  assert(l1GtPsbQuadStringToEnum("GtExternalQ") == GtExternalQ);
  assert(l1GtPsbQuadStringToEnum("PsbQuadNull") == PsbQuadNull);
  assert(l1GtPsbQuadStringToEnum("FooBar") == PsbQuadNull);

  assert(l1GtPsbQuadEnumToString(Free) == "Free");
  assert(l1GtPsbQuadEnumToString(TechTr) == "TechTr");
  assert(l1GtPsbQuadEnumToString(IsoEGQ) == "IsoEGQ");
  assert(l1GtPsbQuadEnumToString(NoIsoEGQ) == "NoIsoEGQ");
  assert(l1GtPsbQuadEnumToString(CenJetQ) == "CenJetQ");
  assert(l1GtPsbQuadEnumToString(ForJetQ) == "ForJetQ");
  assert(l1GtPsbQuadEnumToString(TauJetQ) == "TauJetQ");
  assert(l1GtPsbQuadEnumToString(ESumsQ) == "ESumsQ");
  assert(l1GtPsbQuadEnumToString(JetCountsQ) == "JetCountsQ");
  assert(l1GtPsbQuadEnumToString(MQB1) == "MQB1");
  assert(l1GtPsbQuadEnumToString(MQB2) == "MQB2");
  assert(l1GtPsbQuadEnumToString(MQF3) == "MQF3");
  assert(l1GtPsbQuadEnumToString(MQF4) == "MQF4");
  assert(l1GtPsbQuadEnumToString(MQB5) == "MQB5");
  assert(l1GtPsbQuadEnumToString(MQB6) == "MQB6");
  assert(l1GtPsbQuadEnumToString(MQF7) == "MQF7");
  assert(l1GtPsbQuadEnumToString(MQF8) == "MQF8");
  assert(l1GtPsbQuadEnumToString(MQB9) == "MQB9");
  assert(l1GtPsbQuadEnumToString(MQB10) == "MQB10");
  assert(l1GtPsbQuadEnumToString(MQF11) == "MQF11");
  assert(l1GtPsbQuadEnumToString(MQF12) == "MQF12");
  assert(l1GtPsbQuadEnumToString(CastorQ) == "CastorQ");
  assert(l1GtPsbQuadEnumToString(HfQ) == "HfQ");
  assert(l1GtPsbQuadEnumToString(BptxQ) == "BptxQ");
  assert(l1GtPsbQuadEnumToString(GtExternalQ) == "GtExternalQ");
  assert(l1GtPsbQuadEnumToString(PsbQuadNull) == "PsbQuadNull");
  assert(l1GtPsbQuadEnumToString((L1GtPsbQuad)-1) == "PsbQuadNull");

  assert(l1GtConditionTypeStringToEnum("TypeNull") == TypeNull);
  assert(l1GtConditionTypeStringToEnum("Type1s") == Type1s);
  assert(l1GtConditionTypeStringToEnum("Type2s") == Type2s);
  assert(l1GtConditionTypeStringToEnum("Type2wsc") == Type2wsc);
  assert(l1GtConditionTypeStringToEnum("Type2cor") == Type2cor);
  assert(l1GtConditionTypeStringToEnum("Type3s") == Type3s);
  assert(l1GtConditionTypeStringToEnum("Type4s") == Type4s);
  assert(l1GtConditionTypeStringToEnum("TypeETM") == TypeETM);
  assert(l1GtConditionTypeStringToEnum("TypeETT") == TypeETT);
  assert(l1GtConditionTypeStringToEnum("TypeHTT") == TypeHTT);
  assert(l1GtConditionTypeStringToEnum("TypeHTM") == TypeHTM);
  assert(l1GtConditionTypeStringToEnum("TypeJetCounts") == TypeJetCounts);
  assert(l1GtConditionTypeStringToEnum("TypeCastor") == TypeCastor);
  assert(l1GtConditionTypeStringToEnum("TypeHfBitCounts") == TypeHfBitCounts);
  assert(l1GtConditionTypeStringToEnum("TypeHfRingEtSums") == TypeHfRingEtSums);
  assert(l1GtConditionTypeStringToEnum("TypeBptx") == TypeBptx);
  assert(l1GtConditionTypeStringToEnum("TypeExternal") == TypeExternal);
  assert(l1GtConditionTypeStringToEnum("FooBar") == TypeNull);

  assert(l1GtConditionTypeEnumToString(TypeNull) == "TypeNull");
  assert(l1GtConditionTypeEnumToString((L1GtConditionType)-10) == "TypeNull");
  assert(l1GtConditionTypeEnumToString(Type1s) == "Type1s");
  assert(l1GtConditionTypeEnumToString(Type2s) == "Type2s");
  assert(l1GtConditionTypeEnumToString(Type2wsc) == "Type2wsc");
  assert(l1GtConditionTypeEnumToString(Type2cor) == "Type2cor");
  assert(l1GtConditionTypeEnumToString(Type3s) == "Type3s");
  assert(l1GtConditionTypeEnumToString(Type4s) == "Type4s");
  assert(l1GtConditionTypeEnumToString(TypeETM) == "TypeETM");
  assert(l1GtConditionTypeEnumToString(TypeETT) == "TypeETT");
  assert(l1GtConditionTypeEnumToString(TypeHTT) == "TypeHTT");
  assert(l1GtConditionTypeEnumToString(TypeHTM) == "TypeHTM");
  assert(l1GtConditionTypeEnumToString(TypeJetCounts) == "TypeJetCounts");
  assert(l1GtConditionTypeEnumToString(TypeCastor) == "TypeCastor");
  assert(l1GtConditionTypeEnumToString(TypeHfBitCounts) == "TypeHfBitCounts");
  assert(l1GtConditionTypeEnumToString(TypeHfRingEtSums) == "TypeHfRingEtSums");
  assert(l1GtConditionTypeEnumToString(TypeBptx) == "TypeBptx");
  assert(l1GtConditionTypeEnumToString(TypeExternal) == "TypeExternal");

  assert(l1GtConditionCategoryStringToEnum("CondNull") == CondNull);
  assert(l1GtConditionCategoryStringToEnum("FooBar") == CondNull);
  assert(l1GtConditionCategoryStringToEnum("CondMuon") == CondMuon);
  assert(l1GtConditionCategoryStringToEnum("CondCalo") == CondCalo);
  assert(l1GtConditionCategoryStringToEnum("CondEnergySum") == CondEnergySum);
  assert(l1GtConditionCategoryStringToEnum("CondJetCounts") == CondJetCounts);
  assert(l1GtConditionCategoryStringToEnum("CondCorrelation") == CondCorrelation);
  assert(l1GtConditionCategoryStringToEnum("CondCastor") == CondCastor);
  assert(l1GtConditionCategoryStringToEnum("CondHfBitCounts") == CondHfBitCounts);
  assert(l1GtConditionCategoryStringToEnum("CondHfRingEtSums") == CondHfRingEtSums);
  assert(l1GtConditionCategoryStringToEnum("CondBptx") == CondBptx);
  assert(l1GtConditionCategoryStringToEnum("CondExternal") == CondExternal);

  assert(l1GtConditionCategoryEnumToString(CondNull) == "CondNull");
  assert(l1GtConditionCategoryEnumToString((L1GtConditionCategory)-10) == "CondNull");
  assert(l1GtConditionCategoryEnumToString(CondMuon) == "CondMuon");
  assert(l1GtConditionCategoryEnumToString(CondCalo) == "CondCalo");
  assert(l1GtConditionCategoryEnumToString(CondEnergySum) == "CondEnergySum");
  assert(l1GtConditionCategoryEnumToString(CondJetCounts) == "CondJetCounts");
  assert(l1GtConditionCategoryEnumToString(CondCorrelation) == "CondCorrelation");
  assert(l1GtConditionCategoryEnumToString(CondCastor) == "CondCastor");
  assert(l1GtConditionCategoryEnumToString(CondHfBitCounts) == "CondHfBitCounts");
  assert(l1GtConditionCategoryEnumToString(CondHfRingEtSums) == "CondHfRingEtSums");
  assert(l1GtConditionCategoryEnumToString(CondBptx) == "CondBptx");
  assert(l1GtConditionCategoryEnumToString(CondExternal) == "CondExternal");
}
