		{
			public:
				virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
			private:
				std::map<int, qualityHits> linkAndQual_;
		};

		class BMTFUnpackerInputsNewQual : public Unpacker
		{
			public:
				virtual bool unpack(const Block& block, UnpackerCollections *coll) override;
			private:
				std::map<int, qualityHits> linkAndQual_;
		};

	}
}

// moved to plugins/SealModule.cc
// DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputsOldQual);
// DEFINE_L1T_UNPACKER(l1t::stage2::BMTFUnpackerInputsNewQual);
