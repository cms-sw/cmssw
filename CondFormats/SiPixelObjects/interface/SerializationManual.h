template <typename T>
template <class Archive>
void PixelDCSObject<T>::Item::serialize(Archive & ar, const unsigned int)
{
    ar & BOOST_SERIALIZATION_NVP(name);
    ar & BOOST_SERIALIZATION_NVP(value);
}

namespace cond {
namespace serialization {

// PixelDCSObject<T>::Item is non-deducible
// We need to make specializations explicit, or move
// Item outside PixelDCSObject as its own template.
template <>
struct access<PixelDCSObject<bool>::Item>
{
    static bool equal_(const PixelDCSObject<bool>::Item & first, const PixelDCSObject<bool>::Item & second)
    {
        return true
            and (equal(first.name, second.name))
            and (equal(first.value, second.value))
        ;
    }
};

template <>
struct access<PixelDCSObject<float>::Item>
{
    static bool equal_(const PixelDCSObject<float>::Item & first, const PixelDCSObject<float>::Item & second)
    {
        return true
            and (equal(first.name, second.name))
            and (equal(first.value, second.value))
        ;
    }
};

template <>
struct access<PixelDCSObject<CaenChannel>::Item>
{
    static bool equal_(const PixelDCSObject<CaenChannel>::Item & first, const PixelDCSObject<CaenChannel>::Item & second)
    {
        return true
            and (equal(first.name, second.name))
            and (equal(first.value, second.value))
        ;
    }
};

}
}

