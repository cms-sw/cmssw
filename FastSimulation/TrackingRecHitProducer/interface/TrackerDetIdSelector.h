#ifndef FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H
#define FastSimulation_TrackingRecHitProducer_TrackerDetIdSelector_H

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#define BOOST_RESULT_OF_USE_DECLTYPE
#define BOOST_SPIRIT_USE_PHOENIX_V3

#include <boost/config/warning_disable.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_rule.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/spirit/include/phoenix.hpp>


#include <iostream>
#include <cstdlib>
#include <typeinfo>
#include <functional>


class TrackerDetIdSelector
{
    private:
        const DetId& _detId;
        const TrackerTopology& _trackerTopology;

    public:
        TrackerDetIdSelector(const DetId& detId, const TrackerTopology& trackerTopology):
            _detId(detId),
            _trackerTopology(trackerTopology)
        {
            namespace qi = boost::spirit::qi;
            namespace ascii = boost::spirit::ascii;
            namespace phoenix = boost::phoenix;
        }


        bool passSelection(std::string selectionStr) const
        {
            namespace qi = boost::spirit::qi;
            namespace ascii = boost::spirit::ascii;
            namespace phoenix = boost::phoenix;

            auto printStr = [] (const int& t, qi::unused_type, qi::unused_type)
            {
                std::cout <<"str: "<<t << std::endl;
            };

            std::string::const_iterator begin = selectionStr.cbegin();
            std::string::const_iterator end = selectionStr.cend();

            qi::rule<std::string::const_iterator, int(), ascii::space_type>
                identifier =
                    '!' >> identifier[qi::_val=!qi::_1] |
                    (qi::true_[qi::_val=1] | qi::false_[qi::_val=0]) |
                    (qi::int_[qi::_val=qi::_1]) |
                    (qi::lexeme[+qi::alpha])[qi::_val=0];

            qi::rule<std::string::const_iterator, int(), ascii::space_type>
                expression =
                    (identifier >> qi::lit(">") >> identifier)[qi::_val=qi::_1>qi::_2] |
                    (identifier >> qi::lit(">=") >> identifier)[qi::_val=qi::_1>=qi::_2] |
                    (identifier >> qi::lit("<") >> identifier)[qi::_val=qi::_1<=qi::_2] |
                    (identifier >> qi::lit("<=") >> identifier)[qi::_val=qi::_1<=qi::_2] |
                    (identifier >> qi::lit("==") >> identifier)[qi::_val=qi::_1==qi::_2] |
                    (identifier >> qi::lit("!=") >> identifier)[qi::_val=qi::_1!=qi::_2] |
                    (identifier >> qi::lit("&") >> identifier)[qi::_val=qi::_1 & qi::_2] |
                    (identifier >> qi::lit("|") >> identifier)[qi::_val=qi::_1 | qi::_2];

            qi::rule<std::string::const_iterator, int(), ascii::space_type>
                combo =
                    ('(' >> combo >> ')' >> qi::lit(">") >> combo)[qi::_val=qi::_1>qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit(">=") >> combo)[qi::_val=qi::_1>=qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("<") >> combo)[qi::_val=qi::_1<qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("<=") >> combo)[qi::_val=qi::_1<=qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("==") >> combo)[qi::_val=qi::_1==qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("!=") >> combo)[qi::_val=qi::_1!=qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("&") >> combo)[qi::_val=qi::_1&qi::_2] |
                    ('(' >> combo >> ')' >> qi::lit("|") >> combo)[qi::_val=qi::_1|qi::_2] |
                    (qi::lit("!") >> '(' >> combo[qi::_val=!qi::_1] >> ')') |
                    ('(' >> combo[qi::_val=qi::_1] >> ')') |
                    expression[qi::_val=qi::_1] |
                    identifier[qi::_val=qi::_1];

            bool success = qi::phrase_parse(begin,end, combo[printStr], ascii::space);
            if (begin!=end)
            {
                std::cout<<"error while parsing:"<<std::endl;
                for (auto it=selectionStr.cbegin(); it!=begin; ++it)
                {
                    std::cout << *it;
                }
                std::cout << "^^^";
                for (auto it=begin; it!=selectionStr.cend(); ++it)
                {
                    std::cout << *it;
                }
                std::cout<<std::endl;
            }

            return success;
        }




        unsigned int pxbLayer(const DetId &id) const
        {
            return 1;
        }
        /*
        unsigned int tobLayer(const DetId &id) const
        {
        }

        unsigned int tibLayer(const DetId &id) const
        {
        }

        unsigned int pxbLadder(const DetId &id) const
        {
        }

        unsigned int pxbModule(const DetId &id) const
        {
        }

        unsigned int pxfModule(const DetId &id) const
        {
        }

        unsigned int tobModule(const DetId &id) const
        {
        }

        unsigned int tecModule(const DetId &id) const
        {
        }

        unsigned int tibModule(const DetId &id) const
        {
        }

        unsigned int tidModule(const DetId &id) const
        {
        }

        unsigned int tobSide(const DetId &id) const
        {
        }

        unsigned int tecSide(const DetId &id) const
        {
        }

        unsigned int tibSide(const DetId &id) const
        {
        }

        unsigned int tidSide(const DetId &id) const
        {
        }

        unsigned int pxfSide(const DetId &id) const
        {
        }

        unsigned int tobRod(const DetId &id) const
        {
        }

        unsigned int tecWheel(const DetId &id) const
        {
        }

        unsigned int tidWheel(const DetId &id) const
        {
        }

        unsigned int tecOrder(const DetId &id) const
        {
        }

        unsigned int tibOrder(const DetId &id) const
        {
        }

        unsigned int tidOrder(const DetId &id) const
        {
        }

        unsigned int tecRing(const DetId &id) const
        {
        }

        unsigned int tidRing(const DetId &id) const
        {
        }

        unsigned int tecPetalNumber(const DetId &id) const
        {
        }


        bool tobIsDoubleSide(const DetId &id) const
        {
        }

        bool tecIsDoubleSide(const DetId &id) const
        {
        }

        bool tibIsDoubleSide(const DetId &id) const
        {
        }

        bool tidIsDoubleSide(const DetId &id) const
        {
        }

        bool tobIsZPlusSide(const DetId &id) const
        {
        }

        bool tobIsZMinusSide(const DetId &id) const
        {
        }

        bool tibIsZPlusSide(const DetId &id) const
        {
        }

        bool tibIsZMinusSide(const DetId &id) const
        {
        }

        bool tidIsZPlusSide(const DetId &id) const
        {
        }

        bool tidIsZMinusSide(const DetId &id) const
        {
        }

        bool tecIsZPlusSide(const DetId &id) const
        {
        }

        bool tecIsZMinusSide(const DetId &id) const
        {
        }

        bool tobIsStereo(const DetId &id) const
        {
        }

        bool tecIsStereo(const DetId &id) const
        {
        }

        bool tibIsStereo(const DetId &id) const
        {
        }

        bool tidIsStereo(const DetId &id) const
        {
        }

        uint32_t tobStereo(const DetId &id) const
        {
        }

        uint32_t tibStereo(const DetId &id) const
        {
        }

        uint32_t tidStereo(const DetId &id) const
        {
        }

        uint32_t tecStereo(const DetId &id) const
        {
        }

        uint32_t tibGlued(const DetId &id) const
        {
        }

        uint32_t tecGlued(const DetId &id) const
        {
        }

        uint32_t tobGlued(const DetId &id) const
        {
        }

        uint32_t tidGlued(const DetId &id) const
        {
        }

        bool tobIsRPhi(const DetId &id) const
        {
        }

        bool tecIsRPhi(const DetId &id) const
        {
        }

        bool tibIsRPhi(const DetId &id) const
        {
        }

        bool tidIsRPhi(const DetId &id) const
        {
        }

        bool tecIsBackPetal(const DetId &id) const
        {
        }

        bool tecIsFrontPetal(const DetId &id) const
        {
        }

        unsigned int tibString(const DetId &id) const
        {
        }


        bool tibIsInternalString(const DetId &id) const
        {
        }

        bool tibIsExternalString(const DetId &id) const
        {
        }

        bool tidIsBackRing(const DetId &id) const
        {
        }

        bool tidIsFrontRing(const DetId &id) const
        {
        }

        unsigned int pxfDisk(const DetId &id) const
        {
        }

        unsigned int pxfBlade(const DetId &id) const
        {
        }

        unsigned int pxfPanel(const DetId &id) const
        {
        }
    */
};

#endif
