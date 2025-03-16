# Description

This package is part of a tutorial about writing heterogeneous code in CMSSW.
It contains various modules and data structures.

`HeterogeneousCore/Tutorial/plugins/alpaka/PFJetsSoAProducer.cc` is an
`EDProducer` to convert PFJets to a simplified SoA format. It shows:
  - a heterogeneous `global::EDProducer` running on CPU;
  - SoA data structures;
  - automatic data transfers from host to device.

`HeterogeneousCore/Tutorial/plugins/alpaka/PFJetsSoACorrector.cc` is an
`EDProducer` to apply residual jet corrections on GPU. It shows:
  - an asynchronous `global::EDProducer` running on GPUs;
  - a simple 1D kernel.

`HeterogeneousCore/Tutorial/plugins/alpaka/SoACorrectorESProducer.cc` is an
`ESProducer` for the jet corrections. It shows:
  - a new `EventSetup` record;
  - new “portable” data structures and EventSetup conditions;
  - a heterogeneous `ESProducer`.

HeterogeneousCore/Tutorial/plugins/alpaka/InvariantMassSelector.cc is an
`EDProducer` to find all jet pairs and triplets passing some selection criteria.
It shows:
  - a `stream::SynchronizingEDProducer`;
  - new persistent and local SoA data structures;
  - automatic copy of a configuration object to the GPUs;
  - more complex 2D and 3D kernels.

`HeterogeneousCore/Tutorial/plugins/PFJetsSoAAnalyzer.cc` is an `EDAnalyzer` to
print the N-tuplets. It shows:
  - a traditional `edm::EDAnalyzer` running on CPU;
  - automatic data transfers from device to host.

It also contains `HeterogeneousCore/Tutorial/test/tutorial.py`, a configuration
file to run the full job on GPUs or on CPU.


# License

This package is free software: you can redistribute it and/or modify it under the terms of
the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.

This module is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this
package. If not, see <https://www.gnu.org/licenses/>.
