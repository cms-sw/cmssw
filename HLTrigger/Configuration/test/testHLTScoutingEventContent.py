import unittest

import FWCore.ParameterSet.Config as cms

# Import the definitions
from HLTrigger.Configuration.HLTrigger_EventContent_cff import HLTScouting
from HLTrigger.Configuration.HLTScouting_EventContent_cff import HLTScoutingExtra, HLTScoutingAll

class TestHLTScoutingEventContent(unittest.TestCase):

    def test_hlt_scouting_consistency(self):
        # Convert cms.vstring to Python sets
        scouting = set(HLTScouting.outputCommands)
        extra = set(HLTScoutingExtra.outputCommands)
        allset = set(HLTScoutingAll.outputCommands)

        # Union of current + extra
        combined = scouting | extra

        # Check consistency
        if combined != allset:
            missing = allset - combined
            extra_stuff = combined - allset
            msg = ["HLTScouting consistency check failed:"]
            if missing:
                msg.append(f"  Missing in (HLTScouting + HLTScoutingExtra): {sorted(missing)}")
            if extra_stuff:
                msg.append(f"  Extra in (HLTScouting + HLTScoutingExtra): {sorted(extra_stuff)}")
            self.fail("\n".join(msg))


if __name__ == "__main__":
    unittest.main()
