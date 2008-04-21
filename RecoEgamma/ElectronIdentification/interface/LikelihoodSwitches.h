#ifndef LikelihoodSwitches_H
#define LikelihoodSwitches_H

struct LikelihoodSwitches {

  LikelihoodSwitches () :
    m_useEoverPIn (false) ,
    m_useDeltaEtaIn (false) ,
    m_useDeltaPhiIn (false) ,
    m_useHoverE (false) ,
    m_useE9overE25 (false) ,
    m_useEoverPOut (false) ,
    m_useDeltaPhiOut (false) ,
    m_useDeltaEtaCalo (false) ,
    m_useInvEMinusInvP (false) ,
    m_useBremFraction (false) ,
    m_useSigmaEtaEta (false) ,
    m_useSigmaPhiPhi (false) ,
    m_useShapeFisher (false) {} ;

  bool m_useEoverPIn ;
  bool m_useDeltaEtaIn ;
  bool m_useDeltaPhiIn ;
  bool m_useHoverE ;
  bool m_useE9overE25 ;
  bool m_useEoverPOut ;
  bool m_useDeltaPhiOut ;
  bool m_useDeltaEtaCalo ;
  bool m_useInvEMinusInvP ;
  bool m_useBremFraction ;
  bool m_useSigmaEtaEta ;
  bool m_useSigmaPhiPhi ;
  bool m_useShapeFisher ;

};

#endif // LikelihoodSwitches_H
