#ifndef GENERS_VARPACK_HH_
#define GENERS_VARPACK_HH_

#include "Alignment/Geners/interface/CPP11_config.hh"
#ifdef CPP11_STD_AVAILABLE

#include <string>
#include <cassert>

#include "Alignment/Geners/interface/tupleIO.hh"
#include "Alignment/Geners/interface/VPackIOCycler.hh"

namespace gs {
    //
    // The VarPack template does not process the class ids of the
    // tuple items every time the reading is performed. This can
    // speed up reading of these items. Apply "checkTypeEveryTime(true)"
    // to enable type check for every read if I/O versions for some
    // of the items can change from one read to another.
    //
    template<typename... Args>
    class VarPack : public std::tuple<Args...>
    {
        template<typename Pack, int N> friend struct Private::VPackIOCycler;

    public:
        typedef std::tuple<Args...> Base;

        inline VarPack() : firstRead_(true), checkTypeEveryTime_(false) {}

        inline explicit VarPack(const Args&... args) :
            std::tuple<Args...>(args...),
            firstRead_(true), checkTypeEveryTime_(false) {}

        inline bool checkTypeEveryTime() const {return checkTypeEveryTime_;}

        inline VarPack& checkTypeEveryTime(const bool newValue)
        {
            checkTypeEveryTime_ = newValue;
            return *this;
        }

        inline ClassId classId() const {return ClassId(*this);}

        // For I/O purposes, this class is identical to std::tuple
        static const char* classname()
        {
            static const std::string name(tuple_class_name<Base>("std::tuple"));
            return name.c_str();
        }
        static inline unsigned version() {return 0;}

        inline bool write(std::ostream& of) const
        {
            return write_item(of, *(static_cast<const Base*>(this)), false);
        }

        static inline void restore(const ClassId& id, std::istream& is,
                                   VarPack* pack)
        {
            assert(pack);
            if (pack->firstRead_ || pack->checkTypeEveryTime_)
            {
                static const ClassId packId(ClassId::makeId<VarPack>());
                packId.ensureSameName(id);
                id.templateParameters(&pack->iostack_);
                assert(pack->iostack_.size() == std::tuple_size<Base>::value);
                pack->firstRead_ = false;
            }
            Private::VPackIOCycler<
                VarPack,std::tuple_size<Base>::value>::read(pack, is);
        }

    private:
        std::vector<std::vector<ClassId> > iostack_;
        bool firstRead_;
        bool checkTypeEveryTime_;
    };


    // Function to simplify creation of variable packs
    template<typename... Args>
    inline VarPack<Args...> make_VarPack(Args... args)
    {
        return VarPack<Args...>(args...);
    }
}

#endif // CPP11_STD_AVAILABLE
#endif // GENERS_VARPACK_HH_

