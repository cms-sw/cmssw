#ifndef L1T_PACKER_STAGE2_REGIONALMUONGMTPACKER_H
#define L1T_PACKER_STAGE2_REGIONALMUONGMTPACKER_H

namespace l1t {
   namespace stage2 {
      class RegionalMuonGMTPacker : public Packer {
         public:
            Blocks pack(const edm::Event&, const PackerTokens*) override;
         private:
            typedef std::map<unsigned int, std::vector<uint32_t>> PayloadMap;
            void packTF(const edm::Event&, const edm::EDGetTokenT<RegionalMuonCandBxCollection>&, Blocks&, const std::vector<unsigned int>&);
      };
   }
}

#endif
