//
// A sequence of jet correction metods for FFTJet
//
// I. Volobouev, 08/02/2012
//

#ifndef JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequence_h
#define JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequence_h

#include "JetMETCorrections/FFTJetObjects/interface/FFTJetCorrector.h"

//
//  Both InitialConverter and FinalConverter must publish
//  result_type typedef.
//
//  InitialConverter must have a method with the signature
//  "result_type operator()(const Jet&) const". The
//  variable of the type returned by this method will be
//  propagated through the correction chain. This should
//  be a simple type with copy constructor and assignment
//  operator, for example "double" or "LorentzVector".
//
//  FinalConverter must have a method with the signature
//  "result_type operator()(const Jet&, const T&) const",
//  where T is now the result_type of InitialConverter.
//  It can be a different result type from that defined
//  in InitialConverter. result_type from FinalConverter
//  will be the type of variable returned by the "correct"
//  method of the FFTJetCorrectorSequence class.
//
template
<
    class Jet,
    template<class> class InitialConverter,
    template<class> class FinalConverter
>
class FFTJetCorrectorSequence
{
public:
    typedef Jet jet_type;
    typedef typename InitialConverter<Jet>::result_type adjustable_type;
    typedef typename FinalConverter<Jet>::result_type result_type;
    typedef FFTJetCorrector<jet_type, adjustable_type> Corrector;

    inline FFTJetCorrectorSequence() {}

    inline FFTJetCorrectorSequence(const std::vector<Corrector>& s)
        : sequence_(s) {}

    inline FFTJetCorrectorSequence(const std::vector<Corrector>& s,
                                   const InitialConverter<Jet>& i,
                                   const FinalConverter<Jet>& f)
        : sequence_(s), cinit_(i), cfinal_(f) {}

    inline FFTJetCorrectorSequence(const InitialConverter<Jet>& i,
                                   const FinalConverter<Jet>& f)
        : cinit_(i), cfinal_(f) {}

    inline void addCorrector(const Corrector& c)
        {sequence_.push_back(c);}

    inline unsigned nLevels() const {return sequence_.size();}

    inline const std::vector<Corrector>& getCorrectors() const
        {return sequence_;}

    result_type correct(const Jet& jet, const bool isMC) const
    {
        adjustable_type a1(cinit_(jet));
        adjustable_type a2(a1);
        adjustable_type* first = &a1;
        adjustable_type* second = &a2;

        const unsigned nLevels = sequence_.size();
        for (unsigned level=0; level<nLevels; ++level)
        {
            first = level % 2 ? &a2 : &a1;
            second = level % 2 ? &a1 : &a2;
            sequence_[level].correct(jet, isMC, *first, second);
        }

        return cfinal_(jet, *second);
    }

    const Corrector& operator[](const unsigned i) const
        {return sequence_.at(i);}

private:
    std::vector<Corrector> sequence_;
    InitialConverter<Jet> cinit_;
    FinalConverter<Jet> cfinal_;
};

#endif // JetMETCorrections_FFTJetObjects_FFTJetCorrectorSequence_h
