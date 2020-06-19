# Nix development environment
#
# build:
# nix-build -I "BBPpkgs=https://github.com/BlueBrain/bbp-nixpkgs/archive/master.tar.gz" default.nix
#
# build and test:
# nix-build -I "BBPpkgs=https://goo.gl/wTvE5t" --arg testExec true  default.nix  -j 4
#
# dev shell:
# nix-shell -I "BBPpkgs=https://goo.gl/wTvE5t"  default.nix
#
with import <BBPpkgs> { };

{
	highfive = highfive.overrideDerivation (oldAttr: rec {
      name = "highfive-0.1";
      src = ./.;

	});
}
