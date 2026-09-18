#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""The python reader names levels and vertex roles itself, so that a script without a
CMSSW environment can read a dumped graph. These tests hold those tables against the
headers: a level added in C++ and not here would read as an empty set of levels."""

import json
import os
import re
import tempfile
import unittest

from PhysicsTools.TruthInfo.graphTools import (
    LEVEL_BITS,
    PARTICLE_ROLES,
    VERTEX_REASONS,
    VERTEX_ROLES,
    TruthGraphView,
    bunchCrossingOf,
    eventIndexOf,
    isSignalEventId,
)


def headerText(package, name):
    for base in (os.environ.get("CMSSW_BASE"), os.environ.get("CMSSW_RELEASE_BASE")):
        if not base:
            continue
        path = os.path.join(base, "src", package, "interface", name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                return handle.read()
    raise RuntimeError("cannot find %s/interface/%s" % (package, name))


class TestGraphToolsTables(unittest.TestCase):
    def test_level_bits_match_the_headers(self):
        particleData = headerText("SimDataFormats/TruthInfo", "ParticleData.h")
        bitOfFlag = {
            name: 1 << int(shift) for name, shift in re.findall(r"(\w+) = 1u << (\d+),", particleData)
        }
        self.assertTrue(bitOfFlag, "no LevelFlag values found in ParticleData.h")

        levels = headerText("PhysicsTools/TruthInfo", "TruthLevels.h")
        rows = re.findall(r"\{Level::\w+, LevelFlag::(\w+), \"(\w+)\"\}", levels)
        self.assertEqual(len(rows), len(re.findall(r"\{Level::", levels)))

        expected = {name: bitOfFlag[flag] for flag, name in rows}
        signalName = re.search(r'kSignalLevelName = "(\w+)"', levels).group(1)
        expected[signalName] = bitOfFlag["Signal"]

        self.assertEqual(LEVEL_BITS, expected)

    def test_vertex_roles_match_the_header(self):
        vertexData = headerText("SimDataFormats/TruthInfo", "VertexData.h")
        block = re.search(r"enum class VertexRole : uint8_t \{(.*?)\}", vertexData, re.S).group(1)
        names = [name for name, _ in re.findall(r"(\w+) = (\d+)", block)]
        self.assertEqual(VERTEX_ROLES, names)

    def test_particle_roles_match_the_header(self):
        particleData = headerText("SimDataFormats/TruthInfo", "ParticleData.h")
        block = re.search(r"enum class ParticleRole : uint8_t \{(.*?)\}", particleData, re.S).group(1)
        names = [name for name, _ in re.findall(r"(\w+) = (\d+)", block)]
        self.assertEqual(PARTICLE_ROLES, names)

    def test_vertex_reasons_match_the_header(self):
        vertexData = headerText("SimDataFormats/TruthInfo", "VertexData.h")
        block = re.search(r"enum class VertexReason : uint8_t \{(.*?)\};", vertexData, re.S).group(1)
        names = []
        for line in block.splitlines():
            code = line.split("//")[0].strip().rstrip(",")
            if code:
                names.append(code.split("=")[0].strip())
        self.assertEqual(VERTEX_REASONS, names)


class TestGraphToolsNavigation(unittest.TestCase):
    # The third in-time pileup interaction: bunch crossing 0, index 3.
    PILEUP = 3

    def graph(self):
        particles = [
            dict(id=0, pdgId=111, status=2, statusFlags=0, hasGen=True, hasSim=False, eventId=self.PILEUP,
                 genEvent=3, levelFlags=LEVEL_BITS["reconstructableFinalState"], role="Normal",
                 p4=(0.0, 0.0, 2.0, 2.1)),
            dict(id=1, pdgId=22, status=1, statusFlags=0, hasGen=True, hasSim=True, eventId=self.PILEUP,
                 genEvent=3, levelFlags=LEVEL_BITS["stableDecayProducts"], role="Normal", p4=(0.0, 0.0, 1.2, 1.2)),
            dict(id=2, pdgId=22, status=1, statusFlags=0, hasGen=True, hasSim=True, eventId=self.PILEUP,
                 genEvent=3, levelFlags=LEVEL_BITS["stableDecayProducts"], role="Normal", p4=(0.0, 0.0, 0.8, 0.9)),
            dict(id=3, pdgId=211, status=1, statusFlags=0, hasGen=True, hasSim=True, eventId=0,
                 genEvent=0, levelFlags=LEVEL_BITS["signal"], role="Normal", p4=(0.0, 0.0, 5.0, 5.0)),
        ]
        vertices = [
            dict(id=0, role="Normal", reason="Decay", hasGen=True, hasSim=False, eventId=self.PILEUP,
                 x4=(0.0, 0.0, 0.0, 0.0)),
            dict(id=1, role="Interaction", reason="Unknown", hasGen=False, hasSim=False, eventId=0,
                 x4=(0.0, 0.0, 0.0, 0.0)),
        ]
        return TruthGraphView(
            particles,
            vertices,
            decayVertices=[[0], [], [], []],
            productionVertices=[[], [0], [0], [1]],
            incoming=[[0], []],
            outgoing=[[1, 2], [3]],
        )

    def test_navigation(self):
        graph = self.graph()
        self.assertEqual(graph.children(0), [1, 2])
        self.assertEqual(graph.parents(1), [0])
        self.assertEqual(graph.descendants(0), [1, 2])
        self.assertEqual(graph.children(1), [])
        self.assertTrue(graph.hasMomentum(0))

    def test_child_and_sibling_lookups(self):
        graph = self.graph()
        # The signed id has to match, as in C++.
        self.assertEqual(graph.firstChildWithPdgId(0, 22), 1)
        self.assertIsNone(graph.firstChildWithPdgId(0, -22))
        self.assertIsNone(graph.firstChildWithPdgId(1, 22))
        # The two photons come from one vertex, so each is the other's sibling.
        self.assertEqual(graph.productionSiblings(1), [2])
        self.assertEqual(graph.productionSiblings(2), [1])
        self.assertEqual(graph.productionSiblings(0), [])

    def test_particles_at_levels(self):
        graph = self.graph()
        # particle 0 is reconstructableFinalState, 1 and 2 are stableDecayProducts.
        self.assertEqual(graph.particlesAtLevels(["reconstructableFinalState"]), [0])
        self.assertEqual(
            graph.particlesAtLevels(["reconstructableFinalState", "stableDecayProducts"]), [0, 1, 2])
        self.assertEqual(
            graph.particlesAtLevels(["reconstructableFinalState", "stableDecayProducts"], "all"), [])
        self.assertEqual(graph.particlesAtLevels([]), [])
        self.assertRaises(ValueError, graph.particlesAtLevels, ["signal"], "either")

    def test_signal_particles_and_interactions(self):
        graph = self.graph()
        self.assertEqual(graph.signalParticles(), [3])

        interactions = graph.interactions()
        self.assertEqual(len(interactions), 1)
        # Only the signal carries an Interaction vertex in this fixture.
        self.assertTrue(interactions[0]["isSignal"])
        self.assertEqual(interactions[0]["vertexId"], 1)
        self.assertEqual(interactions[0]["outgoingParticles"], [3])
        self.assertEqual(interactions[0]["position"], (0.0, 0.0, 0.0, 0.0))

    def test_levels_and_provenance(self):
        graph = self.graph()
        self.assertEqual(graph.particlesOfLevel("reconstructableFinalState"), [0])
        self.assertEqual(graph.levels(3), ["signal"])
        self.assertTrue(graph.isFromPileup(0))
        self.assertTrue(graph.isSignal(3))
        self.assertEqual(graph.interactionIds(), [0, self.PILEUP])
        self.assertEqual(graph.particlesOfInteraction(0), [3])
        self.assertEqual(graph.verticesWithRole("Interaction"), [1])
        self.assertIn("Interaction 1", graph.summary())

    def test_from_json(self):
        payload = {
            "run": 1, "lumi": 1, "event": 7,
            "particles": [
                {"id": 0, "pdgId": 111, "status": 2, "statusFlags": 0, "hasGen": True, "hasSim": False,
                 "role": "Normal", "eventId": 3, "bunchCrossing": 0, "eventIndex": 3, "genEvent": 3,
                 "p4": [0.0, 0.0, 2.0, 2.1], "levels": ["reconstructableFinalState"]},
                {"id": 1, "pdgId": 22, "status": 1, "statusFlags": 0, "hasGen": True, "hasSim": False,
                 "role": "Normal", "eventId": 3, "bunchCrossing": 0, "eventIndex": 3, "genEvent": 3,
                 "p4": [None, None, None, None], "levels": []},
            ],
            "vertices": [
                {"id": 0, "role": "Normal", "reason": "Decay", "hasGen": True, "hasSim": False, "eventId": 3,
                 "bunchCrossing": 0, "eventIndex": 3, "x4": [0.0, 0.0, 1.0, 0.0], "in": [0], "out": [1]},
            ],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle)
            path = handle.name
        graph = TruthGraphView.fromJson(path)
        os.unlink(path)
        self.assertEqual(graph.eventId, (1, 1, 7))
        self.assertEqual(graph.children(0), [1])
        self.assertEqual(graph.parents(1), [0])
        self.assertEqual(graph.levels(0), ["reconstructableFinalState"])
        self.assertEqual(graph.vertex(0)["reason"], "Decay")
        # A value the dumper wrote as null reads as no momentum.
        self.assertFalse(graph.hasMomentum(1))
        self.assertTrue(graph.isFromPileup(0))

    def test_last_copy(self):
        # Z(0) -> Z(1) gamma(2) ; Z(1) -> e(3) e(4)
        particles = [
            dict(id=i, pdgId=p, status=s, statusFlags=0, hasGen=True, hasSim=False, eventId=0, genEvent=0,
                 levelFlags=0, role="Normal", p4=(0.0, 0.0, 1.0, 1.0))
            for i, (p, s) in enumerate([(23, 2), (23, 2), (22, 1), (11, 1), (-11, 1)])
        ]
        vertices = [dict(id=0, role="Normal", reason="Decay", hasGen=True, hasSim=False, eventId=0, x4=(0, 0, 0, 0)),
                    dict(id=1, role="Normal", reason="Decay", hasGen=True, hasSim=False, eventId=0, x4=(0, 0, 0, 0))]
        graph = TruthGraphView(particles, vertices, decayVertices=[[0], [1], [], [], []],
                               productionVertices=[[], [0], [0], [1], [1]], incoming=[[0], [1]],
                               outgoing=[[1, 2], [3, 4]])
        self.assertEqual(graph.lastCopy(0), 1)
        self.assertEqual(graph.lastCopy(1), 1)
        self.assertEqual(graph.lastCopy(3), 3)

    def test_event_id_decoding(self):
        self.assertTrue(isSignalEventId(0))
        self.assertEqual(bunchCrossingOf(0), 0)
        self.assertEqual(eventIndexOf(7), 7)
        self.assertFalse(isSignalEventId(7))
        # Bunch crossing 2, interaction 5, and the same one before the signal crossing.
        self.assertEqual(bunchCrossingOf((2 << 16) | 5), 2)
        self.assertEqual(eventIndexOf((2 << 16) | 5), 5)
        self.assertEqual(bunchCrossingOf(0x80000000 | (2 << 16) | 5), -2)


if __name__ == "__main__":
    unittest.main()
