#!/usr/bin/env python3
# Original author: Felice Pantaleo (CERN) <felice.pantaleo@cern.ch>

"""Read and navigate a truth::Graph from python.

The graph is stored as CSR arrays, so a plain probe walks four offset arrays by hand and
decodes the packed interaction id and the level bits itself. This wraps that once.

Two sources, one interface:

    from PhysicsTools.TruthInfo.graphTools import TruthGraphView, eventGraphs

    for graph in eventGraphs("step2.root"):            # every event of an EDM file
        for i in graph.particlesOfLevel("bHadrons"):
            print(graph.pdgId(i), graph.p4(i), graph.children(i))

    graph = TruthGraphView.fromProduct(handle.product())   # inside an FWLite loop
    graph = TruthGraphView.fromJson("truthlogicalgraph_run1_lumi1_event7.json")

The JSON form is what TruthLogicalGraphDumper writes with jsonFile set. It carries the
same numbers at full precision, and needs no CMSSW environment to read, which is what a
plotting or audit script wants.
"""

import json

# The LevelFlag bits of SimDataFormats/TruthInfo/interface/ParticleData.h, and the names
# of PhysicsTools/TruthInfo/interface/TruthLevels.h. A reader with no CMSSW environment
# has no other way to name them, so they are listed here and checked against the release
# by test/graphTools_t.py.
LEVEL_BITS = {
    "stableLegsFromInitialState": 1 << 0,
    "hardProcess": 1 << 1,
    "stableDecayProducts": 1 << 2,
    "caloBoundary": 1 << 3,
    "reconstructableFromSignal": 1 << 5,
    "underlyingEvent": 1 << 6,
    "partonJets": 1 << 7,
    "bHadrons": 1 << 8,
    "cHadrons": 1 << 9,
    "reconstructableFinalState": 1 << 10,
    "tauVisibleHadronic": 1 << 11,
    "tauVisibleLeptonic": 1 << 12,
    # Last, as truth::levelNamesOf reports it: not a level row, the selection's own flag.
    "signal": 1 << 4,
}

VERTEX_ROLES = ["Normal", "InitialState", "UnderlyingEvent", "Interaction", "BeamSideInput"]
PARTICLE_ROLES = ["Normal", "Connector", "SignalStandIn"]
VERTEX_REASONS = [
    "Unknown", "Primary", "Decay", "Bremsstrahlung", "Ionisation", "PairConversion", "Compton",
    "PhotoElectric", "Annihilation", "Rayleigh", "CoulombScattering", "HadronInelastic", "HadronElastic",
    "NuclearCapture", "ChargeExchange", "HadronAtRest", "Other", "HardScatter", "ShowerBranching",
    "Hadronization",
]


def _name(table, code):
    return table[code] if 0 <= code < len(table) else str(code)


def _asInt(value):
    """A uint8_t member reaches python as an int or as a one-character string,
    depending on the ROOT version."""
    return value if isinstance(value, int) else ord(value)


def bunchCrossingOf(eventId):
    """The bunch crossing packed into an EncodedEventId, negative before the signal one."""
    crossing = (eventId >> 16) & 0x7FFF
    return -crossing if eventId & 0x80000000 else crossing


def eventIndexOf(eventId):
    """The index of the interaction inside its bunch crossing, 0 for the signal."""
    return eventId & 0xFFFF


def isSignalEventId(eventId):
    return bunchCrossingOf(eventId) == 0 and eventIndexOf(eventId) == 0


class TruthGraphView:
    """Navigation over one event's graph. Ids are the graph's own particle and vertex ids.

    Each particle and vertex is a dict with the same keys from either source; roles and
    reasons are names. The levels are the flags the producer stamped on the graph, so a
    file written before a level existed reads as no member of it. fromProduct copies the
    whole product into python: measured 1.5 s and 190 MB for a PU200 event, which is what
    an audit script can afford and an event loop cannot.
    """

    def __init__(self, particles, vertices, decayVertices, productionVertices, incoming, outgoing, eventId=None):
        self._particles = particles
        self._vertices = vertices
        self._decayVertices = decayVertices
        self._productionVertices = productionVertices
        self._incoming = incoming
        self._outgoing = outgoing
        self.eventId = eventId
        # What the level rules read from the graph itself; empty unless a source set them.
        self.reconstructablePdgIds = []
        self.signalSeedPdgIds = []
        self.seedHadronFlavors = []

    # --- construction ------------------------------------------------------------
    @classmethod
    def fromProduct(cls, graph, eventId=None):
        """Wrap a truth::Graph as FWLite or PyROOT hands it over."""

        def csr(offsets, flat):
            offsets, flat = list(offsets), list(flat)
            return [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]

        particles = [
            dict(
                id=i,
                pdgId=p.pdgId,
                status=p.status,
                statusFlags=p.statusFlags,
                hasGen=p.genNode >= 0,
                hasSim=p.simNode >= 0,
                eventId=p.eventId,
                genEvent=p.genEvent,
                levelFlags=p.levelFlags,
                role=_name(PARTICLE_ROLES, _asInt(p.role)),
                p4=(p.momentum.px(), p.momentum.py(), p.momentum.pz(), p.momentum.e()),
            )
            for i, p in enumerate(graph.particles())
        ]
        vertices = [
            dict(
                id=i,
                role=_name(VERTEX_ROLES, _asInt(v.role)),
                reason=_name(VERTEX_REASONS, _asInt(v.reason)),
                hasGen=v.genNode >= 0,
                hasSim=v.simNode >= 0,
                eventId=v.eventId,
                x4=(v.position.x(), v.position.y(), v.position.z(), v.position.t()),
            )
            for i, v in enumerate(graph.vertices())
        ]
        view = cls(
            particles,
            vertices,
            csr(graph.particleToDecayVertexOffsets(), graph.particleToDecayVertices()),
            csr(graph.particleToProductionVertexOffsets(), graph.particleToProductionVertices()),
            csr(graph.vertexToIncomingParticleOffsets(), graph.vertexToIncomingParticles()),
            csr(graph.vertexToOutgoingParticleOffsets(), graph.vertexToOutgoingParticles()),
            eventId,
        )
        view.reconstructablePdgIds = list(graph.reconstructablePdgIds())
        view.signalSeedPdgIds = list(graph.signalSeedPdgIds())
        view.seedHadronFlavors = list(graph.seedHadronFlavors())
        return view

    @classmethod
    def fromJson(cls, path):
        """Read what TruthLogicalGraphDumper wrote with jsonFile set."""
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        particles = payload["particles"]
        vertices = payload["vertices"]
        for particle in particles:
            particle["levelFlags"] = sum(LEVEL_BITS.get(name, 0) for name in particle.get("levels", []))
            # The dumper writes null for a value that is not a number.
            particle["p4"] = tuple(0.0 if x is None else x for x in particle["p4"])
        decay = [[] for _ in particles]
        production = [[] for _ in particles]
        incoming, outgoing = [], []
        for vertex in vertices:
            vertex["x4"] = tuple(0.0 if x is None else x for x in vertex["x4"])
            incoming.append(list(vertex["in"]))
            outgoing.append(list(vertex["out"]))
            for particle in vertex["in"]:
                decay[particle].append(vertex["id"])
            for particle in vertex["out"]:
                production[particle].append(vertex["id"])
        eventId = (payload.get("run"), payload.get("lumi"), payload.get("event"))
        view = cls(particles, vertices, decay, production, incoming, outgoing, eventId)
        view.reconstructablePdgIds = list(payload.get("reconstructablePdgIds", []))
        view.signalSeedPdgIds = list(payload.get("signalSeedPdgIds", []))
        view.seedHadronFlavors = list(payload.get("seedHadronFlavors", []))
        return view

    # --- particles ---------------------------------------------------------------
    def nParticles(self):
        return len(self._particles)

    def nVertices(self):
        return len(self._vertices)

    def particle(self, particleId):
        return self._particles[particleId]

    def vertex(self, vertexId):
        return self._vertices[vertexId]

    def pdgId(self, particleId):
        return self._particles[particleId]["pdgId"]

    def p4(self, particleId):
        return self._particles[particleId]["p4"]

    def hasMomentum(self, particleId):
        return self._particles[particleId]["p4"][3] > 0.0

    def lastCopy(self, particleId):
        """The last copy of a radiating chain, as truth::lastCopyOf: follow the one child of
        the same species through the one decay vertex until the species changes."""
        pdgId = self.pdgId(particleId)
        current = particleId
        for _ in range(self.nParticles()):
            if self._particles[current]["status"] == 1:
                break
            decays = self._decayVertices[current]
            if len(decays) != 1:
                break
            same = [c for c in self._outgoing[decays[0]] if self.pdgId(c) == pdgId]
            if len(same) != 1:
                break
            current = same[0]
        return current

    # --- navigation --------------------------------------------------------------
    def children(self, particleId):
        return [c for v in self._decayVertices[particleId] for c in self._outgoing[v]]

    def parents(self, particleId):
        return [p for v in self._productionVertices[particleId] for p in self._incoming[v]]

    def decayVertices(self, particleId):
        return self._decayVertices[particleId]

    def productionVertices(self, particleId):
        return self._productionVertices[particleId]

    def incomingParticles(self, vertexId):
        return self._incoming[vertexId]

    def outgoingParticles(self, vertexId):
        return self._outgoing[vertexId]

    def descendants(self, particleId):
        """Every particle below this one, each once."""
        seen, stack, out = {particleId}, [particleId], []
        while stack:
            for child in self.children(stack.pop()):
                if child not in seen:
                    seen.add(child)
                    out.append(child)
                    stack.append(child)
        return out

    def firstChildWithPdgId(self, particleId, pdgId):
        """The first child with exactly this signed pdgId, as Particle::firstChildWithPdgId.
        A radiating particle carries its decay products on its last copy."""
        for child in self.children(particleId):
            if child != particleId and self.pdgId(child) == pdgId:
                return child
        return None

    def productionSiblings(self, particleId):
        """The other particles produced where this one was, each once: what recoils
        against it."""
        out = []
        for vertexId in self._productionVertices[particleId]:
            for sibling in self._outgoing[vertexId]:
                if sibling != particleId and sibling not in out:
                    out.append(sibling)
        return out

    # --- levels and provenance ---------------------------------------------------
    def levels(self, particleId):
        flags = self._particles[particleId]["levelFlags"]
        return [name for name, bit in LEVEL_BITS.items() if flags & bit]

    def isAtLevel(self, particleId, level):
        return bool(self._particles[particleId]["levelFlags"] & LEVEL_BITS[level])

    def particlesOfLevel(self, level):
        bit = LEVEL_BITS[level]
        return [i for i, p in enumerate(self._particles) if p["levelFlags"] & bit]

    def bunchCrossing(self, particleId):
        return bunchCrossingOf(self._particles[particleId]["eventId"])

    def eventIndex(self, particleId):
        return eventIndexOf(self._particles[particleId]["eventId"])

    def isSignal(self, particleId):
        return isSignalEventId(self._particles[particleId]["eventId"])

    def isFromPileup(self, particleId):
        return not self.isSignal(particleId)

    def interactionIds(self):
        """The packed interaction ids present, signal first. interactions() carries the
        same list with the vertex, the position and the products of each one."""
        return sorted({p["eventId"] for p in self._particles})

    def particlesOfInteraction(self, eventId):
        return [i for i, p in enumerate(self._particles) if p["eventId"] == eventId]

    def particlesAtLevels(self, levels, match="any"):
        """The particles several levels name together, in id order and each once.
        match="any" is the union, match="all" the intersection. As particlesOfLevel, this
        reads the stamped flags, where the C++ particlesAtLevels recomputes the antichain
        from the graph."""
        if match not in ("any", "all"):
            raise ValueError("match is 'any' or 'all', not %r" % match)
        bits = {LEVEL_BITS[level] for level in levels}
        if not bits:
            return []
        out = []
        for i, particle in enumerate(self._particles):
            flags = particle["levelFlags"]
            hits = sum(1 for bit in bits if flags & bit)
            keep = hits == len(bits) if match == "all" else hits > 0
            if keep:
                out.append(i)
        return out

    def signalParticles(self):
        """What the selection preset named as the signal. Signal is a stamped flag and not
        a level row, so no preset means an empty list."""
        return self.particlesOfLevel("signal")

    # --- vertices ----------------------------------------------------------------
    def interactionVertices(self):
        """The vertices the graph marks as interaction points, in id order, one per
        interaction. Only a selection preset builds them."""
        return self.verticesWithRole("Interaction")

    def verticesWithRole(self, role):
        return [i for i, v in enumerate(self._vertices) if v["role"] == role]

    def interactions(self):
        """One entry per overlaid interaction, the signal first and the pile-up after it by
        bunch crossing then by index. Each entry carries the vertex that stands for the
        interaction point, its position and what came out of it. Ask isSignal rather than
        taking the first entry on faith: a pile-up-only graph holds no signal."""
        found = {}
        for vertexId, vertex in enumerate(self._vertices):
            if vertex["role"] == "Interaction":
                found.setdefault(vertex["eventId"], vertexId)
        placeholder = set()
        if not found:
            # No preset ran, so elect the lowest-numbered usable production vertex of each
            # interaction, as truth::interactions does.
            elected, onlyPlaceholders = {}, {}
            for particleId, particle in enumerate(self._particles):
                vertices = self._productionVertices[particleId]
                if not vertices:
                    continue
                vertexId = vertices[0]
                vertex = self._vertices[vertexId]
                usable = vertex["hasSim"] or any(c != 0.0 for c in vertex["x4"])
                target = elected if usable else onlyPlaceholders
                eventId = particle["eventId"]
                target[eventId] = min(target.get(eventId, vertexId), vertexId)
            found = dict(elected)
            for eventId, vertexId in onlyPlaceholders.items():
                if eventId not in found:
                    found[eventId] = vertexId
                    placeholder.add(eventId)

        out = []
        for eventId, vertexId in found.items():
            out.append(dict(eventId=eventId,
                            vertexId=vertexId,
                            isPlaceholder=eventId in placeholder,
                            isSignal=isSignalEventId(eventId),
                            bunchCrossing=bunchCrossingOf(eventId),
                            eventIndex=eventIndexOf(eventId),
                            position=self._vertices[vertexId]["x4"],
                            outgoingParticles=list(self._outgoing[vertexId])))
        out.sort(key=lambda i: (not i["isSignal"], i["bunchCrossing"], i["eventIndex"]))
        return out

    def summary(self):
        """One line per interaction: particles, vertices and artificial roles."""
        particles = {}
        roles = {}
        for particle in self._particles:
            particles[particle["eventId"]] = particles.get(particle["eventId"], 0) + 1
        for vertex in self._vertices:
            if vertex["role"] != "Normal":
                key = (vertex["eventId"], vertex["role"])
                roles[key] = roles.get(key, 0) + 1
        lines = []
        for eventId in sorted(particles):
            named = ", ".join("%s %d" % (role, n) for (eid, role), n in sorted(roles.items()) if eid == eventId)
            lines.append("eid %d (bx %d, index %d): %d particles, %s"
                         % (eventId, bunchCrossingOf(eventId), eventIndexOf(eventId), particles[eventId],
                            named or "no artificial vertex"))
        return "\n".join(lines)


def eventGraphs(fileName, label="truthLogicalGraphProducer", maxEvents=-1):
    """Yield one TruthGraphView per event of an EDM file, through FWLite."""
    import ROOT

    ROOT.gROOT.SetBatch(True)
    ROOT.gSystem.Load("libFWCoreFWLite.so")
    ROOT.FWLiteEnabler.enable()
    from DataFormats.FWLite import Events, Handle

    handle = Handle("truth::Graph")
    for i, event in enumerate(Events(fileName)):
        if 0 <= maxEvents <= i:
            break
        event.getByLabel(label, handle)
        auxiliary = event.eventAuxiliary()
        yield TruthGraphView.fromProduct(
            handle.product(), (auxiliary.run(), auxiliary.luminosityBlock(), auxiliary.event())
        )
