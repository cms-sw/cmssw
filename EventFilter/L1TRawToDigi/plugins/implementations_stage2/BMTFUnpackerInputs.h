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
