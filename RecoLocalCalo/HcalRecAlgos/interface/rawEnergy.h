#ifndef RecoLocalCalo_HcalRecAlgos_rawEnergy_h
#define RecoLocalCalo_HcalRecAlgos_rawEnergy_h

#include "Alignment/Geners/interface/IOIsClassType.hh"

namespace HcalRecAlgosPrivate {
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

    template<typename T, bool is_class_type=gs::IOIsClassType<T>::value>
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

    template<typename T, bool is_class_type=gs::IOIsClassType<T>::value>
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

#endif // RecoLocalCalo_HcalRecAlgos_rawEnergy_h
