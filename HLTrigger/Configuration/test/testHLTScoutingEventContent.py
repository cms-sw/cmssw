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

            msg = []
            msg.append("ERROR: HLTScouting consistency check failed!")
            msg.append("")
            msg.append("The combined content of HLTScouting + HLTScoutingExtra")
            msg.append("does not match HLTScoutingAll.")
            msg.append("")
            if missing:
                msg.append(" Missing expected entries in (HLTScouting + HLTScoutingExtra):")
                for item in sorted(missing):
                    msg.append(f"    - '{item}'")
            if extra_stuff:
                msg.append(" Extra entries in (HLTScouting + HLTScoutingExtra):")
                for item in sorted(extra_stuff):
                    msg.append(f"    - '{item}'")
            msg.append("")
            msg.append(
                "To fix this: edit "
                "HLTrigger/Configuration/python/HLTScouting_EventContent_cff.py"
            )
            msg.append("   (do NOT edit HLTrigger_EventContent_cff.py, it is auto-generated).")
            self.fail("\n".join(msg))


if __name__ == "__main__":
    unittest.main()
