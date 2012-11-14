#ifndef NPSTAT_STORABLEHISTONDFUNCTOR_HH_
#define NPSTAT_STORABLEHISTONDFUNCTOR_HH_

/*!
// \file StorableHistoNDFunctor.h
//
// \brief Storable multivariate functor which uses histogram contents for
//        data representation
//
// Author: I. Volobouev
//
// July 2012
*/

#include "JetMETCorrections/InterpolationTables/interface/SimpleFunctors.h"
#include "JetMETCorrections/InterpolationTables/interface/StorableMultivariateFunctor.h"

#include "JetMETCorrections/InterpolationTables/interface/HistoND.h"

namespace npstat {
    /**
    // This class adapts HistoND template together with various histogram
    // interpolation functions to the StorableMultivariateFunctor interface
    */
    template
    <
        class Numeric,
        class Axis = HistoAxis,
        class Converter = Same<Numeric>
    >
    class StorableHistoNDFunctor : public StorableMultivariateFunctor
    {
        template <typename Num2, typename Axis2, typename Conv2>
        friend class StorableHistoNDFunctor;

    public:
        typedef HistoND<Numeric,Axis> Table;

        //@{
        /**
        // Constructor from a pre-existing histogram. The "degree"
        // argument specifies the interpolation degree which can be
        // 0, 1, or 3.
        */
        template <class Num2>
        inline StorableHistoNDFunctor(
            const HistoND<Num2,Axis>& table, const unsigned degree=1)
            : StorableMultivariateFunctor(), table_(table), deg_(degree)
            {validateInterDegree(degree, table.isUniformlyBinned());}

        template <class Num2>
        inline StorableHistoNDFunctor(
            const HistoND<Num2,Axis>& table,
            const unsigned degree,
            const std::string& descr)
            : StorableMultivariateFunctor(descr), table_(table), deg_(degree)
            {validateInterDegree(degree, table.isUniformlyBinned());}
        //@}

        /** Converting copy constructor */
        template <class Num2, class Conv2>
        inline StorableHistoNDFunctor(
            const StorableHistoNDFunctor<Num2,Axis,Conv2>& tab)
            : StorableMultivariateFunctor(tab.description()),
              table_(tab.table_, Same<Num2>(), tab.title().c_str(),
                     tab.accumulatedDataLabel().c_str()), deg_(tab.deg_) {}

        virtual ~StorableHistoNDFunctor() {}

        virtual unsigned minDim() const {return table_.dim();};

        virtual double operator()(const double* point, unsigned dim) const;

        /** Retrieve interpolation degree */
        inline unsigned interpolationDegree() const {return deg_;}

        /** Set interpolation degree (0, 1, or 3) */
        void setInterpolationDegree(const unsigned deg);

        //@{
        /** Retrieve the underlying HistoND object */
        inline Table& interpolator() {return table_;}
        inline const Table& interpolator() const {return table_;}
        //@}

        //@{
        /** Retrieve the tabulated data */
        inline ArrayND<Numeric>& table()
            {return const_cast<ArrayND<Numeric>&>(table_.binContents());}

        inline const ArrayND<Numeric>& table() const
            {return table_.binContents();}
        //@}

        /** Change the coordinate converter */
        inline void setConverter(const Converter& conv) {conv_ = conv;}

        //@{
        // Method related to "geners" I/O
        virtual gs::ClassId classId() const {return gs::ClassId(*this);}
        virtual bool write(std::ostream& of) const;
        //@}

        // I/O methods needed for reading
        static inline const char* classname();
        static inline unsigned version() {return 1;}
        static StorableHistoNDFunctor* read(
            const gs::ClassId& id, std::istream& in);

    protected:
        virtual bool isEqual(const StorableMultivariateFunctor& other) const
        {
            // Note the use of static_cast rather than dynamic_cast below.
            // static_cast works faster and it is guaranteed to succeed here.
            const StorableHistoNDFunctor& r = 
                static_cast<const StorableHistoNDFunctor&>(other);
            return table_ == r.table_ && deg_ == r.deg_ &&
                   this->description() == other.description();
        }

    private:
        StorableHistoNDFunctor();

        Table table_;
        unsigned deg_;
        Converter conv_;

        static void validateInterDegree(unsigned deg, bool isUniform);
    };
}

#include "JetMETCorrections/InterpolationTables/interface/NpstatException.h"

#include "Alignment/Geners/interface/binaryIO.hh"
#include "Alignment/Geners/interface/CPP11_auto_ptr.hh"
#include "Alignment/Geners/interface/IOException.hh"

#include "JetMETCorrections/InterpolationTables/interface/interpolateHistoND.h"

namespace npstat {
    template <class Numeric, class Axis, class Converter>
    double StorableHistoNDFunctor<Numeric,Axis,Converter>::operator()(
        const double* point, const unsigned dim) const
    {
        return conv_(interpolateHistoND(table_, point, dim, deg_));
    }

    template <class Numeric, class Axis, class Converter>
    const char* StorableHistoNDFunctor<Numeric,Axis,Converter>::classname()
    {
        static const std::string myClass(gs::template_class_name<Numeric,Axis>(
                                            "npstat::StorableHistoNDFunctor"));
        return myClass.c_str();
    }

    template <class Numeric, class Axis, class Converter>
    bool StorableHistoNDFunctor<Numeric,Axis,Converter>::write(std::ostream& of) const
    {
        gs::write_pod(of, this->description());
        gs::write_pod(of, deg_);
        return table_.classId().write(of) && table_.write(of);
    }

    template <class Numeric, class Axis, class Converter>
    StorableHistoNDFunctor<Numeric,Axis,Converter>* 
    StorableHistoNDFunctor<Numeric,Axis,Converter>::read(
        const gs::ClassId& id, std::istream& in)
    {
        static const gs::ClassId current(
            gs::ClassId::makeId<StorableHistoNDFunctor<Numeric,Axis> >());
        current.ensureSameId(id);

        std::string descr;
        gs::read_pod(in, &descr);
        unsigned deg;
        gs::read_pod(in, &deg);
        gs::ClassId tabid(in, 1);
        if (in.fail()) throw gs::IOReadFailure(
            "In npstat::StorableHistoNDFunctor::read: "
            "input stream failure");
        CPP11_auto_ptr<Table> tab(Table::read(tabid, in));
        return new StorableHistoNDFunctor(*tab, deg, descr);
    }

    template <class Numeric, class Axis, class Converter>
    inline void StorableHistoNDFunctor<Numeric,Axis,Converter>::setInterpolationDegree(
        const unsigned deg) 
    {
        validateInterDegree(deg, table_.isUniformlyBinned());
        deg_ = deg;
    }

    template <class Numeric, class Axis, class Converter>
    inline void StorableHistoNDFunctor<Numeric,Axis,Converter>::validateInterDegree(
        const unsigned deg, const bool isUniform)
    {
        const bool ok = isUniform ? (deg == 0 || deg == 1 || deg == 3) :
                                    (deg == 0 || deg == 1);
        if (!ok)
            throw npstat::NpstatInvalidArgument(
                "In npstat::StorableHistoNDFunctor::validateInterDegree: "
                "unsupported interpolation degree");
    }
}


#endif // NPSTAT_STORABLEHISTONDFUNCTOR_HH_

