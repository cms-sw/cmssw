//=========================================================================
// ProcessItem.hh
//
// We will employ a compile-time visitor pattern. The idea
// is to replace function calls (which are difficult to develop)
// with partial template specializations (which are relatively
// easy to develop).
//
// The call signature of the visitor will be
//
// bool InspectingVisitor<Arg1, Arg2, T, Stage>::process(
//         const T&, Arg1&, Arg2*, const bool processClassId);
//
// or
//
// bool ModifyingVisitor<Arg1, Arg2, T, Stage>::process(
//         T&, Arg1&, Arg2*, const bool processClassId);
//
// The processing will be terminated as soon as any call to
// the visitor's process function returns "false".
//
// I. Volobouev
// October 2010
//=========================================================================

#ifndef GENERS_PROCESSITEM_HH_
#define GENERS_PROCESSITEM_HH_

#include "Alignment/Geners/interface/IOTraits.hh"
#include "Alignment/Geners/interface/Int2Type.hh"

namespace gs {
    // Special types to designate stages in container processing
    struct InContainerHeader 
    {
        static const char* stage() {return "InContainerHeader";}
    };

    struct InContainerSize
    {
        static const char* stage() {return "InContainerSize";}
    };

    struct InContainerFooter
    {
        static const char* stage() {return "InContainerFooter";}
    };

    struct InContainerCycle
    {
        static const char* stage() {return "InContainerCycle";}
    };

    struct InPODArray
    {
        static const char* stage() {return "InPODArray";}
    };
}

// I am not aware of an easy way to have both const and non-const
// version of a template defined in the same code fragment. This is
// why you see some preprocessor tricks below -- the alternative
// of maintaining separate const and non-const codes is much worse.

#ifdef GENERS_GENERATE_CONST_IO_PROCESSOR
#undef GENERS_GENERATE_CONST_IO_PROCESSOR
#endif

//
// This is an internal header. Applications should NEVER use it directly.
//
#ifdef GENERS_GENERATED_IO_PROCESSOR
#undef GENERS_GENERATED_IO_PROCESSOR
#endif

#ifdef GENERS_GENERATED_IO_CONSTNESS
#undef GENERS_GENERATED_IO_CONSTNESS
#endif

#ifdef GENERS_CONTAINER_ITERATION_PROC
#undef GENERS_CONTAINER_ITERATION_PROC
#endif

#ifdef GENERS_GENERIC_ITEM_PROCESSOR 
#undef GENERS_GENERIC_ITEM_PROCESSOR 
#endif

#ifdef GENERS_GENERATE_CONST_IO_PROCESSOR
#define GENERS_GENERATED_IO_PROCESSOR ProcessItemLVL1
#define GENERS_GENERATED_IO_CONSTNESS const
#define GENERS_CONTAINER_ITERATION_PROC iterate_const_container
#define GENERS_GENERIC_ITEM_PROCESSOR process_const_item
#else
#define GENERS_GENERATED_IO_PROCESSOR ProcessItemLVL2
#define GENERS_GENERATED_IO_CONSTNESS
#define GENERS_CONTAINER_ITERATION_PROC iterate_container
#define GENERS_GENERIC_ITEM_PROCESSOR process_item
#endif

namespace gs {
    namespace Private {
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2,
            int Mode
        >
        struct GENERS_GENERATED_IO_PROCESSOR ;
    }
}

namespace gs {
    template
    <
        template <typename, typename, typename, typename> class Visitor,
        typename T,
        typename Arg1,
        typename Arg2
    >
    bool GENERS_GENERIC_ITEM_PROCESSOR (GENERS_GENERATED_IO_CONSTNESS T& obj,
                               Arg1& a1, Arg2* p2, const bool processClassId)
    {
        // The following works like a compile-time "switch" statement.
        // This switch is too high level to be implemented using types,
        // so we must do something like what is coded below.
        typedef IOTraits<T> M;
        return Private:: GENERS_GENERATED_IO_PROCESSOR <Visitor, T, Arg1, Arg2,
            M::Signature & (M::ISPOD | 
                            M::ISSTDCONTAINER | 
                            M::ISWRITABLE | 
                            M::ISPOINTER |
                            M::ISSHAREDPTR |
                            M::ISIOPTR |
                            M::ISPAIR |
                            M::ISSTRING |
                            M::ISTUPLE |
                            M::ISEXTERNAL
                )>::process(obj, a1, p2, processClassId);
    }
}

namespace gs {
    namespace Private {
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_CONTAINER_ITERATION_PROC
        {
#ifdef GENERS_GENERATE_CONST_IO_PROCESSOR
            static bool process(const T& v, Arg1& a1, Arg2* p2, std::size_t)
            {
                bool itemStatus = true;
                typename T::const_iterator end = v.end();
                for (typename T::const_iterator it = v.begin();
                     it != end && itemStatus; ++it)
                    itemStatus = process_const_item<Visitor>(*it,a1,p2,false);
                return itemStatus;
            }
#else
            static bool process(T& obj, Arg1& a1, Arg2* p2,
                                const std::size_t newSize)
            {
                bool itemStatus = true;
                for (std::size_t i=0; i<newSize && itemStatus; ++i)
                    itemStatus = Visitor<Arg1,Arg2,T,InContainerCycle>::process(
                        obj, a1, p2, i);
                return itemStatus;
            }
#endif
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPOD>
        {
            // POD-processing visitor
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPOD> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISWRITABLE>
        {
            // Processor of writable objects
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISWRITABLE> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPOINTER>
        {
            // There is not enough info here to decide how the pointer
            // should be managed. Therefore, just call the visitor.
            // However, turn on the class id writing because pointer
            // usually signifies polymorphic class.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, bool /* processClassId */)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPOINTER> >::process(
                    obj, a1, p2, true);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISIOPTR>
        {
            // There is not enough info here to decide how the pointer
            // should be managed. Therefore, just call the visitor.
            // In this particular case, we are passing the "processClassId"
            // value as is.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,typename T::element_type*,
                               Int2Type<IOTraits<int>::ISPOINTER> >::process(
                                   obj.getIOReference(), a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSHAREDPTR>
        {
            // There is not enough info here to decide how the shared pointer
            // should be managed. Therefore, just call the visitor.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, bool /* processClassId */)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISSHAREDPTR> >::process(
                    obj, a1, p2, true);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPAIR>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPAIR> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISTUPLE>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISTUPLE> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISEXTERNAL>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISEXTERNAL> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSTRING>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISSTRING> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        // Processing of containers which do not support "write",
        // "read" or "restore"
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        class GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSTDCONTAINER>
        {
        private:
            // The following function will be fired on containers which
            // store PODs contiguously
            static bool process2(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                 Arg2* p2, const bool processClassId, Int2Type<true>)
            {
                return Visitor<Arg1,Arg2,T,InPODArray>::process(
                    obj, a1, p2, processClassId);
            }

            // The following function will be fired on containers which
            // do not not store PODs contiguously
            static bool process2(GENERS_GENERATED_IO_CONSTNESS T& v, Arg1& a1,
                                 Arg2* p2, const bool processClassId, Int2Type<false>)
            {
                GENERS_GENERATED_IO_CONSTNESS std::size_t sz = v.size();
                return Visitor<Arg1,Arg2,T,InContainerSize>::process(
                    sz, a1, p2, processClassId) && 
                    GENERS_CONTAINER_ITERATION_PROC<Visitor,T,Arg1,Arg2>::process(
                        v, a1, p2, sz) && 
                    sz == v.size();
            }

        public:
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                if (Visitor<Arg1,Arg2,T,InContainerHeader>::process(
                        obj, a1, p2, processClassId))
                {
                    // Do generic container processing in case the container
                    // either does not have contiguous storage or its items
                    // are non-PODs. Otherwise do fast processing.
                    bool bodyStatus = true, footerStatus = true;
                    try
                    {
                        bodyStatus = process2(obj, a1, p2, processClassId,
                                 Int2Type<IOTraits<typename T::value_type>::IsPOD &&
                                 IOTraits<T>::IsContiguous>());
                    }
                    catch (...) 
                    {
                        footerStatus = Visitor<Arg1,Arg2,T,InContainerFooter>::process(
                            obj, a1, p2, processClassId);
                        throw;
                    }
                    footerStatus = Visitor<Arg1,Arg2,T,InContainerFooter>::process(
                        obj, a1, p2, processClassId);
                    return bodyStatus && footerStatus;
                }
                else
                    return false;
            }
        };
    }
}

#undef GENERS_GENERATED_IO_PROCESSOR
#undef GENERS_GENERATED_IO_CONSTNESS
#undef GENERS_CONTAINER_ITERATION_PROC
#undef GENERS_GENERIC_ITEM_PROCESSOR 


#define GENERS_GENERATE_CONST_IO_PROCESSOR

//
// This is an internal header. Applications should NEVER use it directly.
//
#ifdef GENERS_GENERATED_IO_PROCESSOR
#undef GENERS_GENERATED_IO_PROCESSOR
#endif

#ifdef GENERS_GENERATED_IO_CONSTNESS
#undef GENERS_GENERATED_IO_CONSTNESS
#endif

#ifdef GENERS_CONTAINER_ITERATION_PROC
#undef GENERS_CONTAINER_ITERATION_PROC
#endif

#ifdef GENERS_GENERIC_ITEM_PROCESSOR 
#undef GENERS_GENERIC_ITEM_PROCESSOR 
#endif

#ifdef GENERS_GENERATE_CONST_IO_PROCESSOR
#define GENERS_GENERATED_IO_PROCESSOR ProcessItemLVL1
#define GENERS_GENERATED_IO_CONSTNESS const
#define GENERS_CONTAINER_ITERATION_PROC iterate_const_container
#define GENERS_GENERIC_ITEM_PROCESSOR process_const_item
#else
#define GENERS_GENERATED_IO_PROCESSOR ProcessItemLVL2
#define GENERS_GENERATED_IO_CONSTNESS
#define GENERS_CONTAINER_ITERATION_PROC iterate_container
#define GENERS_GENERIC_ITEM_PROCESSOR process_item
#endif

namespace gs {
    namespace Private {
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2,
            int Mode
        >
        struct GENERS_GENERATED_IO_PROCESSOR ;
    }
}

namespace gs {
    template
    <
        template <typename, typename, typename, typename> class Visitor,
        typename T,
        typename Arg1,
        typename Arg2
    >
    bool GENERS_GENERIC_ITEM_PROCESSOR (GENERS_GENERATED_IO_CONSTNESS T& obj,
                               Arg1& a1, Arg2* p2, const bool processClassId)
    {
        // The following works like a compile-time "switch" statement.
        // This switch is too high level to be implemented using types,
        // so we must do something like what is coded below.
        typedef IOTraits<T> M;
        return Private:: GENERS_GENERATED_IO_PROCESSOR <Visitor, T, Arg1, Arg2,
            M::Signature & (M::ISPOD | 
                            M::ISSTDCONTAINER | 
                            M::ISWRITABLE | 
                            M::ISPOINTER |
                            M::ISSHAREDPTR |
                            M::ISIOPTR |
                            M::ISPAIR |
                            M::ISSTRING |
                            M::ISTUPLE |
                            M::ISEXTERNAL
                )>::process(obj, a1, p2, processClassId);
    }
}

namespace gs {
    namespace Private {
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_CONTAINER_ITERATION_PROC
        {
#ifdef GENERS_GENERATE_CONST_IO_PROCESSOR
            static bool process(const T& v, Arg1& a1, Arg2* p2, std::size_t)
            {
                bool itemStatus = true;
                typename T::const_iterator end = v.end();
                for (typename T::const_iterator it = v.begin();
                     it != end && itemStatus; ++it)
                    itemStatus = process_const_item<Visitor>(*it,a1,p2,false);
                return itemStatus;
            }
#else
            static bool process(T& obj, Arg1& a1, Arg2* p2,
                                const std::size_t newSize)
            {
                bool itemStatus = true;
                for (std::size_t i=0; i<newSize && itemStatus; ++i)
                    itemStatus = Visitor<Arg1,Arg2,T,InContainerCycle>::process(
                        obj, a1, p2, i);
                return itemStatus;
            }
#endif
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPOD>
        {
            // POD-processing visitor
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPOD> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISWRITABLE>
        {
            // Processor of writable objects
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISWRITABLE> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPOINTER>
        {
            // There is not enough info here to decide how the pointer
            // should be managed. Therefore, just call the visitor.
            // However, turn on the class id writing because pointer
            // usually signifies polymorphic class.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, bool /* processClassId */)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPOINTER> >::process(
                    obj, a1, p2, true);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISIOPTR>
        {
            // There is not enough info here to decide how the pointer
            // should be managed. Therefore, just call the visitor.
            // In this particular case, we are passing the "processClassId"
            // value as is.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,typename T::element_type*,
                               Int2Type<IOTraits<int>::ISPOINTER> >::process(
                                   obj.getIOReference(), a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSHAREDPTR>
        {
            // There is not enough info here to decide how the shared pointer
            // should be managed. Therefore, just call the visitor.
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, bool /* processClassId */)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISSHAREDPTR> >::process(
                    obj, a1, p2, true);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISPAIR>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISPAIR> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISTUPLE>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISTUPLE> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISEXTERNAL>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISEXTERNAL> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        struct GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSTRING>
        {
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                return Visitor<Arg1,Arg2,T,Int2Type<IOTraits<int>::ISSTRING> >::process(
                    obj, a1, p2, processClassId);
            }
        };

        // Processing of containers which do not support "write",
        // "read" or "restore"
        template
        <
            template <typename, typename, typename, typename> class Visitor,
            typename T,
            typename Arg1,
            typename Arg2
        >
        class GENERS_GENERATED_IO_PROCESSOR <Visitor,T,Arg1,Arg2,IOTraits<int>::ISSTDCONTAINER>
        {
        private:
            // The following function will be fired on containers which
            // store PODs contiguously
            static bool process2(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                 Arg2* p2, const bool processClassId, Int2Type<true>)
            {
                return Visitor<Arg1,Arg2,T,InPODArray>::process(
                    obj, a1, p2, processClassId);
            }

            // The following function will be fired on containers which
            // do not not store PODs contiguously
            static bool process2(GENERS_GENERATED_IO_CONSTNESS T& v, Arg1& a1,
                                 Arg2* p2, const bool processClassId, Int2Type<false>)
            {
                GENERS_GENERATED_IO_CONSTNESS std::size_t sz = v.size();
                return Visitor<Arg1,Arg2,T,InContainerSize>::process(
                    sz, a1, p2, processClassId) && 
                    GENERS_CONTAINER_ITERATION_PROC<Visitor,T,Arg1,Arg2>::process(
                        v, a1, p2, sz) && 
                    sz == v.size();
            }

        public:
            static bool process(GENERS_GENERATED_IO_CONSTNESS T& obj, Arg1& a1,
                                Arg2* p2, const bool processClassId)
            {
                if (Visitor<Arg1,Arg2,T,InContainerHeader>::process(
                        obj, a1, p2, processClassId))
                {
                    // Do generic container processing in case the container
                    // either does not have contiguous storage or its items
                    // are non-PODs. Otherwise do fast processing.
                    bool bodyStatus = true, footerStatus = true;
                    try
                    {
                        bodyStatus = process2(obj, a1, p2, processClassId,
                                 Int2Type<IOTraits<typename T::value_type>::IsPOD &&
                                 IOTraits<T>::IsContiguous>());
                    }
                    catch (...) 
                    {
                        footerStatus = Visitor<Arg1,Arg2,T,InContainerFooter>::process(
                            obj, a1, p2, processClassId);
                        throw;
                    }
                    footerStatus = Visitor<Arg1,Arg2,T,InContainerFooter>::process(
                        obj, a1, p2, processClassId);
                    return bodyStatus && footerStatus;
                }
                else
                    return false;
            }
        };
    }
}

#undef GENERS_GENERATED_IO_PROCESSOR
#undef GENERS_GENERATED_IO_CONSTNESS
#undef GENERS_CONTAINER_ITERATION_PROC
#undef GENERS_GENERIC_ITEM_PROCESSOR 


#undef GENERS_GENERATE_CONST_IO_PROCESSOR

#endif // GENERS_PROCESSITEM_HH_

