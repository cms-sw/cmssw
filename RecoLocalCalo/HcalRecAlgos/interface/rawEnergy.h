#ifndef RecoLocalCalo_HcalRecAlgos_rawEnergy_h
#define RecoLocalCalo_HcalRecAlgos_rawEnergy_h

#include "boost/cstdint.hpp"

namespace HcalRecAlgosPrivate {
    template <typename T>
    class IsClassType
    {
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(int C::*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(IsClassType<T>::template test<T>(0)) == 1};
    };

    template <typename T>
    class HasRawEnergySetterHelper
    {
    private:
        template<void (T::*)(float)> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::setRawEnergy>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(HasRawEnergySetterHelper<T>::template test<T>(0)) == 1};
    };

    template<typename T, bool is_class_type=IsClassType<T>::value>
    struct HasRawEnergySetter
    {
        enum {value = false};
    };    

    template<typename T>
    struct HasRawEnergySetter<T, true>
    {
        enum {value = HasRawEnergySetterHelper<T>::value};
    };

    template<typename T, bool>
    struct RawEnergySetter
    {
        inline static void setRawEnergy(T&, float) {}
    };

    template<typename T>
    struct RawEnergySetter<T, true>
    {
        inline static void setRawEnergy(T& h, float e) {h.setRawEnergy(e);}
    };

    template <typename T>
    class HasRawEnergyGetterHelper
    {
    private:
        template<float (T::*)() const> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::eraw>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(HasRawEnergyGetterHelper<T>::template test<T>(0)) == 1};
    };

    template<typename T, bool is_class_type=IsClassType<T>::value>
    struct HasRawEnergyGetter
    {
        enum {value = false};
    };    

    template<typename T>
    struct HasRawEnergyGetter<T, true>
    {
        enum {value = HasRawEnergyGetterHelper<T>::value};
    };

    template<typename T, bool>
    struct RawEnergyGetter
    {
        inline static float getRawEnergy(const T&, float v) {return v;}
    };

    template<typename T>
    struct RawEnergyGetter<T, true>
    {
        inline static float getRawEnergy(const T& h, float) {return h.eraw();}
    };

    template <typename T>
    class HasAuxRecHitGetterHelper
    {
    private:
        template<uint32_t (T::*)() const> struct tester;
        typedef char One;
        typedef struct {char a[2];} Two;
        template<typename C> static One test(tester<&C::auxHBHE>*);
        template<typename C> static Two test(...);

    public:
        enum {value = sizeof(HasAuxRecHitGetterHelper<T>::template test<T>(0)) == 1};
    };

    template<typename T, bool is_class_type=IsClassType<T>::value>
    struct HasAuxRecHitGetter
    {
        enum {value = false};
    };    

    template<typename T>
    struct HasAuxRecHitGetter<T, true>
    {
        enum {value = HasAuxRecHitGetterHelper<T>::value};
    };

    template<typename T, bool>
    struct AuxRecHitGetter
    {
        inline static uint32_t getAuxRecHit(const T&, uint32_t v) {return v;}
    };

    template<typename T>
    struct AuxRecHitGetter<T, true>
    {
        inline static uint32_t getAuxRecHit(const T& h, uint32_t) {return h.auxHBHE();}
    };
}

// Function for setting the raw energy in a code templated
// upon the rechit type. The function call will be ignored
// in case the HcalRecHit type does not have a member function
// "void setRawEnergy(float)".
template <typename HcalRecHit>
inline void setRawEnergy(HcalRecHit& h, float e)
{
    HcalRecAlgosPrivate::RawEnergySetter<HcalRecHit,HcalRecAlgosPrivate::HasRawEnergySetter<HcalRecHit>::value>::setRawEnergy(h, e);
}

// Function for getting the raw energy in a code templated
// upon the rechit type. This function will return "valueIfNoSuchMember"
// in case the HcalRecHit type does not have a member function
// "float eraw() const".
template <typename HcalRecHit>
inline float getRawEnergy(const HcalRecHit& h, float valueIfNoSuchMember=-1.0e20)
{
    return HcalRecAlgosPrivate::RawEnergyGetter<HcalRecHit,HcalRecAlgosPrivate::HasRawEnergyGetter<HcalRecHit>::value>::getRawEnergy(h, valueIfNoSuchMember);
}

// Function for getting the auxiliary word in a code templated
// upon the rechit type. This function will return "valueIfNoSuchMember"
// in case the HcalRecHit type does not have a member function
// "uint32_t auxHBHE() const".
template <typename HcalRecHit>
inline uint32_t getAuxRecHitWord(const HcalRecHit& h, uint32_t valueIfNoSuchMember=4294967295U)
{
    return HcalRecAlgosPrivate::AuxRecHitGetter<HcalRecHit,HcalRecAlgosPrivate::HasAuxRecHitGetter<HcalRecHit>::value>::getAuxRecHit(h, valueIfNoSuchMember);
}

#endif // RecoLocalCalo_HcalRecAlgos_rawEnergy_h
