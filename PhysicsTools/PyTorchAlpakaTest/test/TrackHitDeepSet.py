import torch
import torch.nn as nn


def build_hit_to_track(hit_offsets: torch.Tensor) -> torch.Tensor:
    """
    Construct the mapping once, outside the model.

    hit_offsets has shape [ntracks + 1].

    Hits belonging to track i are:
        [hit_offsets[i], hit_offsets[i + 1])

    Returns:
        hit_to_track: [nhits]
    """
    hit_counts = hit_offsets[1:] - hit_offsets[:-1]

    return torch.repeat_interleave(
        torch.arange(
            hit_counts.numel(),
            dtype=torch.int64,
            device=hit_offsets.device,
        ),
        hit_counts,
    )


class TrackHitDeepSet(nn.Module):
    def __init__(
        self,
        track_feature_dim: int,
        hit_feature_dim: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.track_encoder = nn.Sequential(
            nn.Linear(track_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.hit_encoder = nn.Sequential(
            nn.Linear(hit_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, 1),
        )

    def forward(
        self,
        track_features: torch.Tensor,
        hit_features: torch.Tensor,
        hit_to_track: torch.Tensor,
        track_begin: torch.Tensor,
    ) -> torch.Tensor:
        # One-column SoAs arrive as [N, 1].
        # These indexing operations create views, not copies.
        hit_to_track = hit_to_track[:, 0]  # [nhits]
        track_begin = track_begin[0, 0]    # scalar tensor

        batch_size = track_features.size(0)
        track_end = track_begin + batch_size

        hit_mask = (
            (hit_to_track >= track_begin)
            & (hit_to_track < track_end)
        )  # [nhits]

        batch_hit_features = hit_features[hit_mask]
        batch_hit_to_track = hit_to_track[hit_mask] - track_begin

        hit_embeddings = self.hit_encoder(batch_hit_features)

        pooled_hits = hit_embeddings.new_zeros(
            (batch_size, hit_embeddings.size(1))
        )

        pooled_hits.index_add_(
            0,
            batch_hit_to_track,
            hit_embeddings,
        )

        track_embeddings = self.track_encoder(track_features)

        combined_embeddings = torch.cat(
            (track_embeddings, pooled_hits),
            dim=1,
        )

        return torch.sigmoid(self.classifier(combined_embeddings))

def main() -> int:
    torch.manual_seed(1234)

    ntracks = 8
    track_feature_dim = 3
    hit_feature_dim = 3

    hit_counts = torch.tensor(
        [2, 3, 1, 4, 1, 2, 3, 0],
        dtype=torch.int64,
    )

    hit_offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.int64),
            torch.cumsum(hit_counts, dim=0),
        )
    )

    nhits = int(hit_offsets[-1].item())

    track_features = torch.randn(
        ntracks,
        track_feature_dim,
    )

    hit_features = torch.randn(
        nhits,
        hit_feature_dim,
    )

    # Match the [N, 1] shape produced from a single-column SoA.
    hit_to_track = build_hit_to_track(hit_offsets).unsqueeze(1)

    # Match the [1, 1] shape of the one-element metadata SoA.
    track_begin = torch.tensor(
        [[0]],
        dtype=torch.int64,
    )

    model = TrackHitDeepSet(
        track_feature_dim=track_feature_dim,
        hit_feature_dim=hit_feature_dim,
        embedding_dim=16,
    )

    model.eval()

    with torch.no_grad():
        model_input = (
            track_features,
            hit_features,
            hit_to_track,
            track_begin
        )
        y = model(*model_input)
        print("Input: ")
        for tensor in model_input:
            print(tensor)
        print("Output: ", y)

    tm = torch.jit.trace(model, model_input)
    tm.save("TrackHitDeepSet.pt")

if __name__ == "__main__":
    main()

